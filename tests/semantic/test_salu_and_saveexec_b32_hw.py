from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_and_saveexec_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_and_saveexec_b32 instruction belongs to SOP1 format.
  Instruction:
    s_and_saveexec_b32 sdst, ssrc0

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo) to receive previous EXEC_LO
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = EXEC_LO
    EXEC_LO = ssrc0 & EXEC_LO
    SCC = 1 if EXEC_LO != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
ALL_LANES_ACTIVE_32 = 0xFFFFFFFF
ALL_LANES_INACTIVE_32 = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_and_saveexec_b32(ssrc0: int, old_exec_lo: int) -> tuple[int, int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_and_saveexec_b32 instruction.

    Args:
        ssrc0: 32-bit source operand (uint32)
        old_exec_lo: 32-bit current EXEC_LO mask before instruction execution (uint32)

    Returns:
        tuple (new_exec_lo, sdst, scc):
          - new_exec_lo: updated 32-bit EXEC_LO mask = ssrc0 & old_exec_lo
          - sdst: saved previous EXEC_LO mask = old_exec_lo
          - scc: scalar condition code = 1 if new_exec_lo != 0 else 0
    """
    s0 = u32(ssrc0)
    old_e = u32(old_exec_lo)
    sdst = old_e
    new_exec_lo = u32(s0 & old_e)
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, sdst, scc


class TestSaluAndSaveexecB32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_and_saveexec_b32."""

    def test_oracle_all_ones_exec(self):
        s0 = 0x12345678
        old_exec = ALL_LANES_ACTIVE_32
        new_exec, sdst, scc = ref_and_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, s0)
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_all_zeros_exec(self):
        s0 = 0xFFFFFFFF
        old_exec = ALL_LANES_INACTIVE_32
        new_exec, sdst, scc = ref_and_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, ALL_LANES_INACTIVE_32)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s0_produces_zero_exec(self):
        s0 = 0x0
        old_exec = 0xAAAAAAAA
        new_exec, sdst, scc = ref_and_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, 0xAAAAAAAA)
        self.assertEqual(scc, 0)




class Opcode(Enum):
    S_AND_SAVEEXEC_B32 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC_LO = auto()
    VCC_LO = auto()
    LITERAL32 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 1
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 1) -> ASTOperand:
        if count != 1:
            raise ValueError(f"s_and_saveexec_b32 requires 32-bit register (count=1), got {count}")
        if index < 0 or index > 105:
            raise ValueError(f"SGPR index out of range: s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC_LO, count=1)

    @classmethod
    def vcc_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC_LO, count=1)

    @classmethod
    def literal32(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL32, count=1, value=u32(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=1, value=u32(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.S_AND_SAVEEXEC_B32:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC_LO):
            raise TypeError(f"Destination must be SGPR or EXEC_LO, got {self.dst.kind}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be 32-bit (1 register), got {self.dst.count}")
        if self.src0.count != 1:
            raise ValueError(f"Source must be 32-bit operand, got {self.src0.count}")


class TestSaluAndSaveexecB32AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(4, 1),
            src0=ASTOperand.sgpr(6, 1),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_AND_SAVEEXEC_B32)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)

    def test_ast_rejects_out_of_range_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(106, 1)

    def test_ast_rejects_invalid_count(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(4, 2)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.literal32(0x123),
            src0=ASTOperand.sgpr(6, 1),
        )
        with self.assertRaises(TypeError):
            inst.validate()




class ScalarRegisterState:
    """Simulated state of GCN/RDNA Scalar Registers (SGPRs, EXEC_LO/HI, and SCC)."""

    def __init__(self, exec_lo: int = ALL_LANES_ACTIVE_32, exec_hi: int = ALL_LANES_ACTIVE_32):
        self.sgpr = [0] * 106
        self.exec_lo = u32(exec_lo)
        self.exec_hi = u32(exec_hi)
        self.scc = 0

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index < len(self.sgpr)):
            raise ValueError(f"Invalid SGPR index: s{index}")
        return self.sgpr[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index < len(self.sgpr)):
            raise ValueError(f"Invalid SGPR index: s{index}")
        self.sgpr[index] = u32(val)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        # Read source operand
        if inst.src0.kind == OperandKind.SGPR:
            src_val = self.get_sgpr(inst.src0.index)
        elif inst.src0.kind == OperandKind.EXEC_LO:
            src_val = self.exec_lo
        elif inst.src0.kind in (OperandKind.LITERAL32, OperandKind.INLINE_CONST):
            src_val = inst.src0.value
        else:
            raise ValueError(f"Unsupported operand kind for scalar execution: {inst.src0.kind}")

        new_exec_lo, sdst, scc = ref_and_saveexec_b32(src_val, self.exec_lo)
        if inst.dst.kind == OperandKind.SGPR:
            self.set_sgpr(inst.dst.index, sdst)

        self.exec_lo = new_exec_lo
        self.scc = scc


class TestSaluAndSaveexecB32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_basic_execution(self):
        state = ScalarRegisterState(exec_lo=0xFFFF0000)
        state.set_sgpr(6, 0x12345678)
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(4, 1),
            src0=ASTOperand.sgpr(6, 1),
        )
        state.execute(inst)
        self.assertEqual(state.get_sgpr(4), 0xFFFF0000)
        self.assertEqual(state.exec_lo, 0x12340000)
        self.assertEqual(state.scc, 1)

    def test_scalar_interpreter_literal_source(self):
        state = ScalarRegisterState(exec_lo=0xF0F0F0F0)
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(0, 1),
            src0=ASTOperand.literal32(0x0F0F0F0F),
        )
        state.execute(inst)
        self.assertEqual(state.get_sgpr(0), 0xF0F0F0F0)
        self.assertEqual(state.exec_lo, 0x00000000)
        self.assertEqual(state.scc, 0)




class Wave32SIMDState:
    """Simulated Wave32 multi-lane divergence and active lane state tracking."""

    def __init__(self, exec_mask: int = ALL_LANES_ACTIVE_32):
        self.exec_lo = u32(exec_mask)

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < 32):
            raise IndexError(f"Lane {lane} out of range [0, 31]")
        return bool((self.exec_lo >> lane) & 1)

    def active_lanes_mask(self) -> int:
        return self.exec_lo

    def execute(self, inst: ASTInstruction, scalar_state: ScalarRegisterState) -> None:
        scalar_state.exec_lo = self.exec_lo
        scalar_state.execute(inst)
        self.exec_lo = scalar_state.exec_lo


class TestSaluAndSaveexecB32SIMDDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence, lane deactivation, and translator verification."""

    def test_lanes_deactivated_under_and(self):
        initial_exec = 0xFFFFFFFF
        wave = Wave32SIMDState(exec_mask=initial_exec)
        scalar = ScalarRegisterState(exec_lo=initial_exec)
        scalar.set_sgpr(8, 0x0000FFFF)  # only lanes 0..15 kept

        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(2, 1),
            src0=ASTOperand.sgpr(8, 1),
        )
        wave.execute(inst, scalar)

        for lane in range(0, 16):
            self.assertTrue(wave.is_lane_active(lane), f"Lane {lane} should remain active")
        for lane in range(16, 32):
            self.assertFalse(wave.is_lane_active(lane), f"Lane {lane} should have turned inactive")

    def test_translator_ps_translate_lines(self):
        # Exercises tools/rdna2spirv.py to verify s_and_saveexec_b32 produces correct SPIR-V instructions
        lines = [
            "s_mov_b32 s6, 0x12345678",
            "s_and_saveexec_b32 s4, s6",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_salu_and_saveexec_hw", lines)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitwiseAnd", spv)




class TestSaluAndSaveexecB32Aliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_aliasing_destination_same_as_source(self):
        # s_and_saveexec_b32 s4, s4
        old_exec = 0xAAAAAAAA
        src_init = 0x55555555
        state = ScalarRegisterState(exec_lo=old_exec)
        state.set_sgpr(4, src_init)

        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(4, 1),
            src0=ASTOperand.sgpr(4, 1),
        )
        state.execute(inst)

        self.assertEqual(state.get_sgpr(4), old_exec)
        self.assertEqual(state.exec_lo, src_init & old_exec)

    def test_aliasing_source_is_exec_lo(self):
        # s_and_saveexec_b32 s2, exec_lo
        old_exec = 0x12345678
        state = ScalarRegisterState(exec_lo=old_exec)
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(2, 1),
            src0=ASTOperand.exec_lo(),
        )
        state.execute(inst)
        self.assertEqual(state.get_sgpr(2), old_exec)
        self.assertEqual(state.exec_lo, old_exec)
        self.assertEqual(state.scc, 1)

    def test_untouched_sgprs_preserved(self):
        state = ScalarRegisterState(exec_lo=0x1234)
        for i in range(106):
            state.set_sgpr(i, 0xBEEF0000 + i)

        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(10, 1),
            src0=ASTOperand.sgpr(12, 1),
        )
        state.execute(inst)

        for i in range(106):
            if i == 10:
                self.assertEqual(state.get_sgpr(i), 0x1234)
            else:
                self.assertEqual(state.get_sgpr(i), 0xBEEF0000 + i)




class TestSaluAndSaveexecB32ConditionCodes(unittest.TestCase):
    """Test suite for SCC status flag generation across all boundary scenarios."""

    def test_scc_is_zero_when_result_is_zero(self):
        state = ScalarRegisterState(exec_lo=0x55555555)
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(0, 1),
            src0=ASTOperand.literal32(0xAAAAAAAA),
        )
        state.execute(inst)
        self.assertEqual(state.exec_lo, 0)
        self.assertEqual(state.scc, 0)

    def test_scc_is_one_when_any_bit_active(self):
        state = ScalarRegisterState(exec_lo=ALL_LANES_ACTIVE_32)
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(0, 1),
            src0=ASTOperand.literal32(0x00000001),
        )
        state.execute(inst)
        self.assertEqual(state.exec_lo, 1)
        self.assertEqual(state.scc, 1)

    def test_scc_overwrites_prior_flag_state(self):
        state = ScalarRegisterState(exec_lo=0x0F)
        state.scc = 1
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(0, 1),
            src0=ASTOperand.literal32(0xF0),
        )
        state.execute(inst)
        self.assertEqual(state.scc, 0)

        inst2 = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(0, 1),
            src0=ASTOperand.literal32(0xFF),
        )
        state.exec_lo = 0xFF
        state.execute(inst2)
        self.assertEqual(state.scc, 1)




class TestSaluAndSaveexecB32BitwiseEdgeCases(unittest.TestCase):
    """Test suite for 32-bit IEEE-754 bit representations and boundary bitmasks."""

    def test_ieee754_single_precision_bit_patterns(self):
        cases = [
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC00000,  # Canonical NaN
            0x7F800001,  # Signaling NaN
            0x00000000,  # +0.0
            0x80000000,  # -0.0
            0x00000001,  # Min subnormal
            0x007FFFFF,  # Max subnormal
        ]
        for pattern in cases:
            for old_exec in (0x0, ALL_LANES_ACTIVE_32, 0xAAAAAAAA):
                with self.subTest(pattern=hex(pattern), exec=hex(old_exec)):
                    new_e, d, scc = ref_and_saveexec_b32(pattern, old_exec)
                    self.assertEqual(d, old_exec)
                    self.assertEqual(new_e, pattern & old_exec)
                    self.assertEqual(scc, int(new_e != 0))

    def test_boundary_alternating_bitmasks(self):
        checkerboards = [
            0x55555555,
            0xAAAAAAAA,
            0x33333333,
            0xCCCCCCCC,
            0x0F0F0F0F,
            0xF0F0F0F0,
            0x00FF00FF,
            0xFF00FF00,
        ]
        for cb1 in checkerboards:
            for cb2 in checkerboards:
                new_e, d, scc = ref_and_saveexec_b32(cb1, cb2)
                self.assertEqual(new_e, cb1 & cb2)


if __name__ == "__main__":
    unittest.main()
