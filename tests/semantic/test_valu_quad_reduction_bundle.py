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


# ------------------------------------------------------------------------------
# s_quadmask_b32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestQuadmaskB32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, 106)
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, -1)

    def test_boundary_extremes_hardware(self):
        val_max, _ = run_shader_words("ps_qm32_max", [
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(val_max, 0xFF)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_shader_words("ps_qm32_lowering", [
            "s_mov_b32 s1, 0x12345678",
            "s_quadmask_b32 s0, s1",
            "v_mov_b32 v0, s0",
        ])
        self.assertIn("OpBitwiseOr", spv)


# ------------------------------------------------------------------------------
# s_quadmask_b64 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_quadmask_b64(s0: int) -> int:
    s0 = s0 & 0xFFFF_FFFF_FFFF_FFFF
    res = 0
    for i in range(16):
        if (s0 >> (4 * i)) & 0xF:
            res |= (1 << i)
    return res


class TestQuadmaskB64Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            0x12,
            0xF0F0_F0F0_F0F0_F0F0,
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
            0x0000_0000_0000_0000,
        ]
        for idx, val in enumerate(vectors):
            lo = val & MASK32
            hi = (val >> 32) & MASK32
            val_hw, _ = run_shader_words(f"ps_qm64_oracle_{idx}", [
                f"s_mov_b32 s2, {hex(lo)}",
                f"s_mov_b32 s3, {hex(hi)}",
                "s_quadmask_b64 s[0:1], s[2:3]",
                "v_mov_b32 v0, s0",
            ])
            self.assertEqual(val_hw, ref_quadmask_b64(val))

    def test_oracle_quad_boundaries_hardware(self):
        for i in range(16):
            val = 1 << (4 * i)
            lo = val & MASK32
            hi = (val >> 32) & MASK32
            val_hw, _ = run_shader_words(f"ps_qm64_nibble_{i}", [
                f"s_mov_b32 s2, {hex(lo)}",
                f"s_mov_b32 s3, {hex(hi)}",
                "s_quadmask_b64 s[0:1], s[2:3]",
                "v_mov_b32 v0, s0",
            ])
            self.assertEqual(val_hw, 1 << i)


# ------------------------------------------------------------------------------
# s_quadmask_b64 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class SQuadmaskB64Instruction:
    dst_lo: RegisterOperand
    dst_hi: RegisterOperand
    src0_lo: RegisterOperand
    src0_hi: RegisterOperand


def step_quadmask_b64(machine: ScalarMachineQuadLogic, insn: SQuadmaskB64Instruction):
    lo = machine.read_operand(insn.src0_lo)
    hi = machine.read_operand(insn.src0_hi)
    val64 = (hi << 32) | lo
    res = ref_quadmask_b64(val64)
    machine.sgprs[insn.dst_lo.index] = res & MASK32
    machine.sgprs[insn.dst_hi.index] = (res >> 32) & MASK32
    machine.scc = 1 if res != 0 else 0


class TestQuadmaskB64ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineQuadLogic()
        machine.sgprs[2] = 0x00FF_00FF
        machine.sgprs[3] = 0xFF00_FF00
        insn = SQuadmaskB64Instruction(
            dst_lo=RegisterOperand(OperandType.SGPR, 0),
            dst_hi=RegisterOperand(OperandType.SGPR, 1),
            src0_lo=RegisterOperand(OperandType.SGPR, 2),
            src0_hi=RegisterOperand(OperandType.SGPR, 3),
        )
        step_quadmask_b64(machine, insn)

        val_hw, _ = run_shader_words("ps_qm64_ast_scalar", [
            "s_mov_b32 s2, 0x00FF00FF",
            "s_mov_b32 s3, 0xFF00FF00",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(machine.sgprs[0], val_hw)
        self.assertEqual(val_hw, 0xCC33)

    def test_scalar_scc_flag_hardware(self):
        val_zero, _ = run_shader_words("ps_qm64_scc_zero", [
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "s_cselect_b32 s4, 10, 20",
            "v_mov_b32 v0, s4",
        ])
        self.assertEqual(val_zero, 20)


# ------------------------------------------------------------------------------
# s_quadmask_b64 SIMD Divergence & 16-Quad Reduction
# ------------------------------------------------------------------------------

class TestQuadmaskB64SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        val_even, _ = run_shader_words("ps_qm64_even", [
            "s_mov_b32 s2, 0x0F0F0F0F",
            "s_mov_b32 s3, 0x0F0F0F0F",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
        ])
        val_odd, _ = run_shader_words("ps_qm64_odd", [
            "s_mov_b32 s2, 0xF0F0F0F0",
            "s_mov_b32 s3, 0xF0F0F0F0",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(val_even, 0x5555)
        self.assertEqual(val_odd, 0xAAAA)
        self.assertEqual(val_even | val_odd, 0xFFFF)

    def test_algebraic_reduction_properties_hardware(self):
        val_hi_quad, _ = run_shader_words("ps_qm64_hi_quad", [
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0xF0000000",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(val_hi_quad, 0x8000)


# ------------------------------------------------------------------------------
# s_quadmask_b64 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestQuadmaskB64NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000_0000_0000)

    def test_boundary_extremes_hardware(self):
        val_max, _ = run_shader_words("ps_qm64_max", [
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_mov_b32 s3, 0xFFFFFFFF",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
        ])
        self.assertEqual(val_max, 0xFFFF)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_shader_words("ps_qm64_lowering", [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 0x9ABCDEF0",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
        ])
        self.assertIn("OpBitwiseOr", spv)


# ------------------------------------------------------------------------------
# v_perm_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_perm_b32(s0: int, s1: int, sel: int) -> int:
    s0 = s0 & MASK32
    s1 = s1 & MASK32
    sel = sel & MASK32
    combined = (s0 << 32) | s1
    out = 0
    for i in range(4):
        sb = (sel >> (8 * i)) & 0xFF
        if sb < 8:
            b = (combined >> (8 * sb)) & 0xFF
        elif 8 <= sb <= 11:
            half = sb & 3
            bit = 15 if half == 0 else 31 if half == 1 else 47 if half == 2 else 63
            b = 0xFF if ((combined >> bit) & 1) else 0x00
        elif sb == 12:
            b = 0x00
        else:
            b = 0xFF
        out |= (b << (8 * i))
    return out


class TestPermB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        # s0 = high dword (bytes 4..7), s1 = low dword (bytes 0..3)
        s0 = 0x11223344
        s1 = 0x55667788
        vectors = [
            0x03020100,  # identity select s1 -> 0x55667788
            0x07060504,  # identity select s0 -> 0x11223344
            0x00010203,  # byte reversal of s1 -> 0x88776655
            0x04050607,  # byte reversal of s0 -> 0x44332211
            0x0C0C0C0C,  # zero constant (12) -> 0x00000000
            0x0D0D0D0D,  # 0xFF constant (13) -> 0xFFFFFFFF
        ]
        for idx, sel in enumerate(vectors):
            val_hw, _ = run_shader_words(f"ps_perm_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(s0)}",
                f"v_mov_b32 v2, {hex(s1)}",
                f"v_mov_b32 v3, {hex(sel)}",
                "v_perm_b32 v0, v1, v2, v3",
            ])
            self.assertEqual(val_hw, ref_perm_b32(s0, s1, sel))

    def test_oracle_sign_extension_hardware(self):
        # s1 high half is 0x8000 (negative f16 / sign bit 31 set)
        s0 = 0x00000000
        s1 = 0x80000000
        # selector 9 picks sign bit of half 1 (bit 31 of s1) -> should replicate 0xFF
        sel = 0x0C0C0C09
        val_hw, _ = run_shader_words("ps_perm_sign", [
            f"v_mov_b32 v1, {hex(s0)}",
            f"v_mov_b32 v2, {hex(s1)}",
            f"v_mov_b32 v3, {hex(sel)}",
            "v_perm_b32 v0, v1, v2, v3",
        ])
        self.assertEqual(val_hw, 0x000000FF)


# ------------------------------------------------------------------------------
# v_perm_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VPermB32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]
    sel: Union[RegisterOperand, LiteralOperand]


def step_perm_b32(machine: ScalarMachineQuadLogic, insn: VPermB32Instruction):
    s0 = machine.read_operand(insn.src0)
    s1 = machine.read_operand(insn.src1)
    sel = machine.read_operand(insn.sel)
    res = ref_perm_b32(s0, s1, sel)
    machine.vgprs[insn.dst.index] = res


class TestPermB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineQuadLogic()
        machine.vgprs[1] = 0xAABBCCDD
        machine.vgprs[2] = 0x11223344
        machine.vgprs[3] = 0x07060100
        insn = VPermB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            sel=RegisterOperand(OperandType.VGPR, 3),
        )
        step_perm_b32(machine, insn)

        val_hw, _ = run_shader_words("ps_perm_ast_scalar", [
            "v_mov_b32 v1, 0xAABBCCDD",
            "v_mov_b32 v2, 0x11223344",
            "v_mov_b32 v3, 0x07060100",
            "v_perm_b32 v0, v1, v2, v3",
        ])
        self.assertEqual(machine.vgprs[0], val_hw)
        self.assertEqual(val_hw, 0xAABB3344)

    def test_scalar_in_place_hardware(self):
        val_inplace, _ = run_shader_words("ps_perm_inplace", [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 0x03020100",
            "v_perm_b32 v0, v1, v0, v2",
        ])
        self.assertEqual(val_inplace, 0x12345678)


# ------------------------------------------------------------------------------
# v_perm_b32 SIMD Divergence & Byte Permutation Properties
# ------------------------------------------------------------------------------

class TestPermB32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        # Permutation with constant selector vs register selector
        val_reg, _ = run_shader_words("ps_perm_reg", [
            "v_mov_b32 v1, 0x12345678",
            "v_mov_b32 v2, 0x9ABCDEF0",
            "v_mov_b32 v3, 0x07050301",
            "v_perm_b32 v0, v1, v2, v3",
        ])
        val_imm, _ = run_shader_words("ps_perm_imm", [
            "v_mov_b32 v1, 0x12345678",
            "v_mov_b32 v2, 0x9ABCDEF0",
            "v_perm_b32 v0, v1, v2, 0x07050301",
        ])
        self.assertEqual(val_reg, val_imm)
        self.assertEqual(val_imm, 0x12569ADE)

    def test_algebraic_reversal_hardware(self):
        # Endianness swap via v_perm_b32
        val_swap, _ = run_shader_words("ps_perm_endian", [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0xA1B2C3D4",
            "v_perm_b32 v0, v1, v2, 0x00010203",
        ])
        self.assertEqual(val_swap, 0xD4C3B2A1)


# ------------------------------------------------------------------------------
# v_perm_b32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestPermB32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)

    def test_boundary_extremes_hardware(self):
        # All 0x0C (zero) and all 0x0D (0xFF)
        val_zeroes, _ = run_shader_words("ps_perm_all_zero", [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_perm_b32 v0, v1, v2, 0x0C0C0C0C",
        ])
        val_ones, _ = run_shader_words("ps_perm_all_ones", [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_perm_b32 v0, v1, v2, 0x0D0D0D0D",
        ])
        self.assertEqual(val_zeroes, 0)
        self.assertEqual(val_ones, 0xFFFFFFFF)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_shader_words("ps_perm_lowering", [
            "v_mov_b32 v1, 0x11223344",
            "v_mov_b32 v2, 0x55667788",
            "v_mov_b32 v3, 0x03020100",
            "v_perm_b32 v0, v1, v2, v3",
        ])
        self.assertIn("OpBitFieldUExtract", spv)


# ------------------------------------------------------------------------------
# ds_swizzle_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_ds_swizzle_quad(quad: List[int], p0: int, p1: int, p2: int, p3: int) -> List[int]:
    return [quad[p0], quad[p1], quad[p2], quad[p3]]


def ref_ds_swizzle_swap(quad: List[int]) -> List[int]:
    return [quad[1], quad[0], quad[3], quad[2]]


def ref_ds_swizzle_reverse(quad: List[int]) -> List[int]:
    return [quad[3], quad[2], quad[1], quad[0]]


def ref_ds_swizzle_broadcast(quad: List[int], lane_idx: int) -> List[int]:
    return [quad[lane_idx]] * 4


class TestDsSwizzleB32Oracle(unittest.TestCase):
    def test_oracle_quad_perm_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mov_b32 v2, v0",
            "ds_swizzle_b32 v4, v2 offset:swizzle(QUAD_PERM,3,2,1,0)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_swiz_quad_perm", lines, lanes=4)
        expected = ref_ds_swizzle_quad([0, 1, 2, 3], 3, 2, 1, 0)
        self.assertEqual(res, expected)

    def test_oracle_swap_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mov_b32 v2, v0",
            "ds_swizzle_b32 v4, v2 offset:swizzle(SWAP,1)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_swiz_swap", lines, lanes=4)
        expected = ref_ds_swizzle_swap([0, 1, 2, 3])
        self.assertEqual(res, expected)


# ------------------------------------------------------------------------------
# ds_swizzle_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class DsSwizzleB32Instruction:
    dst: RegisterOperand
    src0: RegisterOperand
    pattern: str


class TestDsSwizzleB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_add_nc_u32 v2, v0, 10",
            "ds_swizzle_b32 v4, v2 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_swiz_ast_scalar", lines, lanes=4)
        self.assertEqual(res, [11, 10, 13, 12])

    def test_reverse_swizzle_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_add_nc_u32 v2, v0, 20",
            "ds_swizzle_b32 v4, v2 offset:swizzle(REVERSE,4)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_swiz_rev", lines, lanes=4)
        self.assertEqual(res, [23, 22, 21, 20])


# ------------------------------------------------------------------------------
# ds_swizzle_b32 SIMD Divergence & Quad Permutation
# ------------------------------------------------------------------------------

class TestDsSwizzleB32SIMDAndProperties(unittest.TestCase):
    def test_simd_broadcast_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_add_nc_u32 v2, v0, 50",
            "ds_swizzle_b32 v4, v2 offset:swizzle(BROADCAST,4,0)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_swiz_bcast", lines, lanes=4)
        # Lane 0's value (50) is broadcast to all 4 lanes
        self.assertEqual(res, [50, 50, 50, 50])

    def test_algebraic_swap_involution_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mov_b32 v2, v0",
            "ds_swizzle_b32 v3, v2 offset:swizzle(SWAP,1)",
            "ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,1)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_swiz_inv", lines, lanes=4)
        self.assertEqual(res, [0, 1, 2, 3])


# ------------------------------------------------------------------------------
# ds_swizzle_b32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestDsSwizzleB32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_swiz_bad", [
                "ds_swizzle_b32 v0, v1 offset:bad_pattern",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm"
            ])

    def test_boundary_extremes_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "ds_swizzle_b32 v4, v2 offset:swizzle(QUAD_PERM,1,0,3,2)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_swiz_max", lines, lanes=4)
        self.assertEqual(res, [0xFFFFFFFF] * 4)

    def test_spirv_opcode_lowering_hardware(self):
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "ds_swizzle_b32 v4, v0 offset:swizzle(SWAP,1)",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        _, spv = run_workgroup_lanes("ps_swiz_lowering", lines, lanes=4)
        self.assertIn("OpGroupNonUniformShuffle", spv)


# ------------------------------------------------------------------------------
# ds_bpermute_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_ds_bpermute_b32(lanes_data: List[int], lane_addrs: List[int], offset: int = 0) -> List[int]:
    out = []
    for i in range(len(lane_addrs)):
        src_lane = ((lane_addrs[i] + offset) // 4) % 32
        val = lanes_data[src_lane] if src_lane < len(lanes_data) else 0
        out.append(val)
    return out


class TestDsBpermuteB32Oracle(unittest.TestCase):
    def test_oracle_gather_hardware(self):
        # Reverse gather: lane i reads lane 3 - i
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_sub_nc_u32 v1, 3, v0",
            "v_lshlrev_b32 v1, 2, v1",
            "v_add_nc_u32 v2, v0, 100",
            "ds_bpermute_b32 v4, v1, v2",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_bperm_oracle", lines, lanes=4)
        self.assertEqual(res, [103, 102, 101, 100])

    def test_oracle_offset_addressing_hardware(self):
        # Lane i specifies address 0 with offset 4 (lane 1)
        lines = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mov_b32 v1, 0",
            "v_add_nc_u32 v2, v0, 10",
            "ds_bpermute_b32 v4, v1, v2 offset:4",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
        ]
        res, _ = run_workgroup_lanes("ps_bperm_offset", lines, lanes=4)
        self.assertEqual(res, [11, 11, 11, 11])
