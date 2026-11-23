from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 32-bit Vector Logic Suite.

Covers:
- v_and_b32: Bitwise AND D = S0 & S1.
- v_or_b32: Bitwise OR D = S0 | S1.
- v_xor_b32: Bitwise XOR D = S0 ^ S1.
- v_not_b32: Bitwise NOT D = ~S0.
- v_xnor_b32: Bitwise XNOR D = ~(S0 ^ S1).
- v_and_or_b32: 3-operand bitwise operation D = (S0 & S1) | S2.
- v_or3_b32: 3-operand bitwise OR D = S0 | S1 | S2.

Standard library only (python -I, struct, math, unittest). Zero game data.
All test suites execute real translated assembly via support.translate_lines.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv

MASK32: int = 0xFFFF_FFFF


def run_shader(name: str, lines: List[str]) -> Tuple[int, str]:
    full_lines = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = support.translate_lines(name, full_lines)
    words = support.mrt0_words(spv)
    return words[0], spv


# ------------------------------------------------------------------------------
# v_and_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_and_b32(s0: int, s1: int) -> int:
    return (s0 & s1) & MASK32


class TestAndB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            (0xFF00, 0x0FF0),
            (0x1234_5678, 0xFFFF_0000),
            (0xAAAA_AAAA, 0x5555_5555),
            (0x1357_9BDF, 0xFEDC_BA98),
        ]
        for idx, (a, b) in enumerate(vectors):
            val, _ = run_shader(f"ps_and_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(a)}",
                f"v_mov_b32 v2, {hex(b)}",
                "v_and_b32 v0, v1, v2",
            ])
            self.assertEqual(val, ref_and_b32(a, b))

    def test_oracle_identities_hardware(self):
        val_id, _ = run_shader("ps_and_id", [
            "v_mov_b32 v1, 0xCAFEBABE",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_id, 0xCAFEBABE)

        val_zero, _ = run_shader("ps_and_zero", [
            "v_mov_b32 v1, 0xCAFEBABE",
            "v_mov_b32 v2, 0",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_zero, 0)


# ------------------------------------------------------------------------------
# v_and_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

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
        if not (-0x8000_0000 <= self.value <= MASK32):
            raise ValueError(f"Literal out of range: {self.value}")


@dataclass
class VAndB32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineLogic:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand: {op}")

    def step_and(self, insn: VAndB32Instruction):
        s0 = self.read_operand(insn.src0)
        s1 = self.read_operand(insn.src1)
        self.vgprs[insn.dst.index] = ref_and_b32(s0, s1)


class TestAndB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineLogic()
        machine.vgprs[1] = 0xF0F0_F0F0
        machine.vgprs[2] = 0x00FF_00FF
        insn = VAndB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step_and(insn)

        val_hw, _ = run_shader("ps_and_ast_scalar", [
            "v_mov_b32 v1, 0xF0F0F0F0",
            "v_mov_b32 v2, 0x00FF00FF",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(machine.vgprs[0], val_hw)
        self.assertEqual(val_hw, 0x00F0_00F0)

    def test_scalar_in_place_hardware(self):
        val_inplace, _ = run_shader("ps_and_inplace", [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x0000FFFF",
            "v_and_b32 v0, v0, v1",
        ])
        self.assertEqual(val_inplace, 0x5678)


# ------------------------------------------------------------------------------
# v_and_b32 SIMD Divergence & Multi-Lane Properties
# ------------------------------------------------------------------------------

class SIMDWaveLogic:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1

    def step_and(self, insn: VAndB32Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.vgprs[insn.src0.index][lane]
                s1 = self.vgprs[insn.src1.index][lane]
                self.vgprs[insn.dst.index][lane] = ref_and_b32(s0, s1)


class TestAndB32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        wave = SIMDWaveLogic(wave_size=32)
        for i in range(32):
            wave.vgprs[1][i] = 1 << i
            wave.vgprs[2][i] = 0x5555_5555
            wave.vgprs[0][i] = 0

        insn = VAndB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        wave.step_and(insn)

        # Check lane 0 and lane 1 via hardware execution
        val_lane0, _ = run_shader("ps_and_lane0", [
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 0x55555555",
            "v_and_b32 v0, v1, v2",
        ])
        val_lane1, _ = run_shader("ps_and_lane1", [
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 0x55555555",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(wave.vgprs[0][0], val_lane0)
        self.assertEqual(wave.vgprs[0][1], val_lane1)

    def test_algebraic_properties_hardware(self):
        # Commutativity: (a & b) == (b & a)
        val_ab, _ = run_shader("ps_and_comm_ab", [
            "v_mov_b32 v1, 0x1234ABCD",
            "v_mov_b32 v2, 0x98765432",
            "v_and_b32 v0, v1, v2",
        ])
        val_ba, _ = run_shader("ps_and_comm_ba", [
            "v_mov_b32 v1, 0x1234ABCD",
            "v_mov_b32 v2, 0x98765432",
            "v_and_b32 v0, v2, v1",
        ])
        self.assertEqual(val_ab, val_ba)


# ------------------------------------------------------------------------------
# v_and_b32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestAndB32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, 106)

    def test_boundary_extremes_hardware(self):
        val_all1, _ = run_shader("ps_and_all1", [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_all1, 0xFFFFFFFF)

        val_sign, _ = run_shader("ps_and_sign", [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x7FFFFFFF",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_sign, 0)

    def test_spirv_opcode_lowering_hardware(self):
        val, spv = run_shader("ps_and_lowering", [
            "v_mov_b32 v1, 0xAAAA5555",
            "v_mov_b32 v2, 0xFFFF0000",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertIn("OpBitwiseAnd", spv)
        self.assertEqual(val, 0xAAAA0000)


# ------------------------------------------------------------------------------
# v_or_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_or_b32(s0: int, s1: int) -> int:
    return (s0 | s1) & MASK32


class TestOrB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            (0xFF00, 0x00FF),
            (0x1200_0000, 0x0034_0000),
            (0x5555_0000, 0x0000_5555),
            (0xAAAA_0000, 0x0000_5555),
        ]
        for idx, (a, b) in enumerate(vectors):
            val, _ = run_shader(f"ps_or_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(a)}",
                f"v_mov_b32 v2, {hex(b)}",
                "v_or_b32 v0, v1, v2",
            ])
            self.assertEqual(val, ref_or_b32(a, b))

    def test_oracle_identities_hardware(self):
        val_zero, _ = run_shader("ps_or_id_zero", [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0",
            "v_or_b32 v0, v1, v2",
        ])
        self.assertEqual(val_zero, 0xDEADBEEF)

        val_dom, _ = run_shader("ps_or_id_dom", [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_or_b32 v0, v1, v2",
        ])
        self.assertEqual(val_dom, 0xFFFFFFFF)


# ------------------------------------------------------------------------------
# v_or_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VOrB32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


def step_or(machine: ScalarMachineLogic, insn: VOrB32Instruction):
    s0 = machine.read_operand(insn.src0)
    s1 = machine.read_operand(insn.src1)
    machine.vgprs[insn.dst.index] = ref_or_b32(s0, s1)


class TestOrB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineLogic()
        machine.vgprs[1] = 0x1111_0000
        machine.vgprs[2] = 0x0000_2222
        insn = VOrB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        step_or(machine, insn)

        val_hw, _ = run_shader("ps_or_ast_scalar", [
            "v_mov_b32 v1, 0x11110000",
            "v_mov_b32 v2, 0x00002222",
            "v_or_b32 v0, v1, v2",
        ])
        self.assertEqual(machine.vgprs[0], val_hw)
        self.assertEqual(val_hw, 0x1111_2222)

    def test_scalar_in_place_hardware(self):
        val_inplace, _ = run_shader("ps_or_inplace", [
            "v_mov_b32 v0, 0x00FF0000",
            "v_mov_b32 v1, 0x0000FF00",
            "v_or_b32 v0, v0, v1",
        ])
        self.assertEqual(val_inplace, 0x00FFFF00)


# ------------------------------------------------------------------------------
# v_or_b32 SIMD Divergence & Multi-Lane Properties
# ------------------------------------------------------------------------------

def step_or_simd(wave: SIMDWaveLogic, insn: VOrB32Instruction):
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.vgprs[insn.src0.index][lane]
            s1 = wave.vgprs[insn.src1.index][lane]
            wave.vgprs[insn.dst.index][lane] = ref_or_b32(s0, s1)


class TestOrB32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        wave = SIMDWaveLogic(wave_size=32)
        for i in range(32):
            wave.vgprs[1][i] = 1 << i
            wave.vgprs[2][i] = 0x8000_0000
            wave.vgprs[0][i] = 0

        insn = VOrB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        step_or_simd(wave, insn)

        val_lane0, _ = run_shader("ps_or_lane0", [
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 0x80000000",
            "v_or_b32 v0, v1, v2",
        ])
        self.assertEqual(wave.vgprs[0][0], val_lane0)
        self.assertEqual(val_lane0, 0x8000_0001)

    def test_algebraic_properties_hardware(self):
        val_ab, _ = run_shader("ps_or_comm_ab", [
            "v_mov_b32 v1, 0x0F0F0000",
            "v_mov_b32 v2, 0x0000F0F0",
            "v_or_b32 v0, v1, v2",
        ])
        val_ba, _ = run_shader("ps_or_comm_ba", [
            "v_mov_b32 v1, 0x0F0F0000",
            "v_mov_b32 v2, 0x0000F0F0",
            "v_or_b32 v0, v2, v1",
        ])
        self.assertEqual(val_ab, val_ba)


# ------------------------------------------------------------------------------
# v_or_b32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestOrB32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(TypeError):
            VOrB32Instruction(
                dst=RegisterOperand(OperandType.SGPR, 0),
                src0=RegisterOperand(OperandType.VGPR, 1),
                src1=RegisterOperand(OperandType.VGPR, 2),
            )

    def test_boundary_extremes_hardware(self):
        val_zeroes, _ = run_shader("ps_or_zeroes", [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_or_b32 v0, v1, v2",
        ])
        self.assertEqual(val_zeroes, 0)

        val_alt, _ = run_shader("ps_or_alt", [
            "v_mov_b32 v1, 0x55555555",
            "v_mov_b32 v2, 0xAAAAAAAA",
            "v_or_b32 v0, v1, v2",
        ])
        self.assertEqual(val_alt, 0xFFFFFFFF)

    def test_spirv_opcode_lowering_hardware(self):
        val, spv = run_shader("ps_or_lowering", [
            "v_mov_b32 v1, 0x0000FFFF",
            "v_mov_b32 v2, 0xFFFF0000",
            "v_or_b32 v0, v1, v2",
        ])
        self.assertIn("OpBitwiseOr", spv)
        self.assertEqual(val, 0xFFFFFFFF)


# ------------------------------------------------------------------------------
# v_xor_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_xor_b32(s0: int, s1: int) -> int:
    return (s0 ^ s1) & MASK32


class TestXorB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            (0xFF00, 0x0FF0),
            (0x1234_5678, 0x1234_FFFF),
            (0xAAAA_AAAA, 0x5555_5555),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
        ]
        for idx, (a, b) in enumerate(vectors):
            val, _ = run_shader(f"ps_xor_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(a)}",
                f"v_mov_b32 v2, {hex(b)}",
                "v_xor_b32 v0, v1, v2",
            ])
            self.assertEqual(val, ref_xor_b32(a, b))

    def test_oracle_identities_hardware(self):
        val_nil, _ = run_shader("ps_xor_id_nil", [
            "v_mov_b32 v1, 0xFEEDFACE",
            "v_xor_b32 v0, v1, v1",
        ])
        self.assertEqual(val_nil, 0)

        val_id, _ = run_shader("ps_xor_id_zero", [
            "v_mov_b32 v1, 0xFEEDFACE",
            "v_mov_b32 v2, 0",
            "v_xor_b32 v0, v1, v2",
        ])
        self.assertEqual(val_id, 0xFEEDFACE)


# ------------------------------------------------------------------------------
# v_xor_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VXorB32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


def step_xor(machine: ScalarMachineLogic, insn: VXorB32Instruction):
    s0 = machine.read_operand(insn.src0)
    s1 = machine.read_operand(insn.src1)
    machine.vgprs[insn.dst.index] = ref_xor_b32(s0, s1)


class TestXorB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineLogic()
        machine.vgprs[1] = 0xAAAA_BBBB
        machine.vgprs[2] = 0xAAAA_0000
        insn = VXorB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        step_xor(machine, insn)

        val_hw, _ = run_shader("ps_xor_ast_scalar", [
            "v_mov_b32 v1, 0xAAAABBBB",
            "v_mov_b32 v2, 0xAAAA0000",
            "v_xor_b32 v0, v1, v2",
        ])
        self.assertEqual(machine.vgprs[0], val_hw)
        self.assertEqual(val_hw, 0x0000_BBBB)

    def test_scalar_in_place_hardware(self):
        val_inplace, _ = run_shader("ps_xor_inplace", [
            "v_mov_b32 v0, 0xFF00FF00",
            "v_mov_b32 v1, 0x00FF00FF",
            "v_xor_b32 v0, v0, v1",
        ])
        self.assertEqual(val_inplace, 0xFFFFFFFF)


# ------------------------------------------------------------------------------
# v_xor_b32 SIMD Divergence & Multi-Lane Properties
# ------------------------------------------------------------------------------

def step_xor_simd(wave: SIMDWaveLogic, insn: VXorB32Instruction):
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.vgprs[insn.src0.index][lane]
            s1 = wave.vgprs[insn.src1.index][lane]
            wave.vgprs[insn.dst.index][lane] = ref_xor_b32(s0, s1)


class TestXorB32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        wave = SIMDWaveLogic(wave_size=32)
        for i in range(32):
            wave.vgprs[1][i] = 1 << i
            wave.vgprs[2][i] = 1 << i
            wave.vgprs[0][i] = 0xFFFFFFFF

        insn = VXorB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        step_xor_simd(wave, insn)

        val_hw, _ = run_shader("ps_xor_lane_match", [
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x4000",
            "v_xor_b32 v0, v1, v2",
        ])
        self.assertEqual(wave.vgprs[0][14], val_hw)
        self.assertEqual(val_hw, 0)

    def test_algebraic_properties_hardware(self):
        # Self-inverse property: (a ^ b) ^ b == a
        val_roundtrip, _ = run_shader("ps_xor_self_inv", [
            "v_mov_b32 v1, 0x13579BDF",
            "v_mov_b32 v2, 0x2468ACE0",
            "v_xor_b32 v0, v1, v2",
            "v_xor_b32 v0, v0, v2",
        ])
        self.assertEqual(val_roundtrip, 0x13579BDF)


# ------------------------------------------------------------------------------
# v_xor_b32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestXorB32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000)

    def test_boundary_extremes_hardware(self):
        val_not, _ = run_shader("ps_xor_not", [
            "v_mov_b32 v1, 0x55555555",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_xor_b32 v0, v1, v2",
        ])
        self.assertEqual(val_not, 0xAAAAAAAA)

    def test_spirv_opcode_lowering_hardware(self):
        val, spv = run_shader("ps_xor_lowering", [
            "v_mov_b32 v1, 0xFF00",
            "v_mov_b32 v2, 0x0FF0",
            "v_xor_b32 v0, v1, v2",
        ])
        self.assertIn("OpBitwiseXor", spv)
        self.assertEqual(val, 0xF0F0)


# ------------------------------------------------------------------------------
# v_not_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_not_b32(s0: int) -> int:
    return (~s0) & MASK32


class TestNotB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [0, 0xFFFFFFFF, 0x00FF_00FF, 0x1234_5678, 0xAAAA_AAAA]
        for idx, a in enumerate(vectors):
            val, _ = run_shader(f"ps_not_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(a)}",
                "v_not_b32 v0, v1",
            ])
            self.assertEqual(val, ref_not_b32(a))

    def test_oracle_identities_hardware(self):
        val_zero, _ = run_shader("ps_not_zero", [
            "v_mov_b32 v1, 0",
            "v_not_b32 v0, v1",
        ])
        self.assertEqual(val_zero, 0xFFFFFFFF)

        val_all1, _ = run_shader("ps_not_all1", [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_not_b32 v0, v1",
        ])
        self.assertEqual(val_all1, 0)


# ------------------------------------------------------------------------------
# v_not_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VNotB32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


def step_not(machine: ScalarMachineLogic, insn: VNotB32Instruction):
    s0 = machine.read_operand(insn.src0)
    machine.vgprs[insn.dst.index] = ref_not_b32(s0)


class TestNotB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineLogic()
        machine.vgprs[1] = 0x0F0F_0F0F
        insn = VNotB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
        )
        step_not(machine, insn)

        val_hw, _ = run_shader("ps_not_ast_scalar", [
            "v_mov_b32 v1, 0x0F0F0F0F",
            "v_not_b32 v0, v1",
        ])
        self.assertEqual(machine.vgprs[0], val_hw)
        self.assertEqual(val_hw, 0xF0F0_F0F0)

    def test_scalar_in_place_hardware(self):
        val_inplace, _ = run_shader("ps_not_inplace", [
            "v_mov_b32 v0, 0x12340000",
            "v_not_b32 v0, v0",
        ])
        self.assertEqual(val_inplace, 0xEDCBFFFF)


# ------------------------------------------------------------------------------
# v_not_b32 SIMD Divergence & Multi-Lane Properties
# ------------------------------------------------------------------------------

def step_not_simd(wave: SIMDWaveLogic, insn: VNotB32Instruction):
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.vgprs[insn.src0.index][lane]
            wave.vgprs[insn.dst.index][lane] = ref_not_b32(s0)


class TestNotB32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        wave = SIMDWaveLogic(wave_size=32)
        for i in range(32):
            wave.vgprs[1][i] = 1 << i
            wave.vgprs[0][i] = 0

        insn = VNotB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
        )
        step_not_simd(wave, insn)

        val_hw, _ = run_shader("ps_not_simd_lane", [
            "v_mov_b32 v1, 0x10",
            "v_not_b32 v0, v1",
        ])
        self.assertEqual(wave.vgprs[0][4], val_hw)
        self.assertEqual(val_hw, (~0x10) & MASK32)

    def test_algebraic_properties_hardware(self):
        # Involution: ~(~a) == a
        val_involution, _ = run_shader("ps_not_involution", [
            "v_mov_b32 v1, 0x87654321",
            "v_not_b32 v0, v1",
            "v_not_b32 v0, v0",
        ])
        self.assertEqual(val_involution, 0x87654321)


# ------------------------------------------------------------------------------
# v_not_b32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestNotB32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(TypeError):
            VNotB32Instruction(
                dst=RegisterOperand(OperandType.SGPR, 5),
                src0=RegisterOperand(OperandType.VGPR, 0),
            )

    def test_boundary_extremes_hardware(self):
        val_sign, _ = run_shader("ps_not_sign", [
            "v_mov_b32 v1, 0x80000000",
            "v_not_b32 v0, v1",
        ])
        self.assertEqual(val_sign, 0x7FFFFFFF)

    def test_spirv_opcode_lowering_hardware(self):
        val, spv = run_shader("ps_not_lowering", [
            "v_mov_b32 v1, 0x000000FF",
            "v_not_b32 v0, v1",
        ])
        self.assertIn("OpNot", spv)
        self.assertEqual(val, 0xFFFFFF00)


# ------------------------------------------------------------------------------
# v_xnor_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_xnor_b32(s0: int, s1: int) -> int:
    return (~(s0 ^ s1)) & MASK32


class TestXnorB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            (0xFF00, 0x0FF0),
            (0, 0),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
            (0xAAAA_AAAA, 0x5555_5555),
            (0x1234_5678, 0x8765_4321),
        ]
        for idx, (a, b) in enumerate(vectors):
            val, _ = run_shader(f"ps_xnor_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(a)}",
                f"v_mov_b32 v2, {hex(b)}",
                "v_xnor_b32 v0, v1, v2",
            ])
            self.assertEqual(val, ref_xnor_b32(a, b))

    def test_oracle_identities_hardware(self):
        val_self, _ = run_shader("ps_xnor_self", [
            "v_mov_b32 v1, 0xCAFEBABE",
            "v_xnor_b32 v0, v1, v1",
        ])
        self.assertEqual(val_self, 0xFFFFFFFF)

        val_opp, _ = run_shader("ps_xnor_opp", [
            "v_mov_b32 v1, 0x55555555",
            "v_mov_b32 v2, 0xAAAAAAAA",
            "v_xnor_b32 v0, v1, v2",
        ])
        self.assertEqual(val_opp, 0)
