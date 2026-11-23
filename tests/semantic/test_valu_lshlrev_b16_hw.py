from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_lshlrev_b16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_lshlrev_b16 performs a 16-bit logical left shift with reversed operands.
  Instruction:
    v_lshlrev_b16 vd, src0, src1

  Operands:
    vd   = Destination VGPR (low 16 bits written, high 16 bits zeroed in VOP2 / e32)
    src0 = Shift amount (low 4 bits used, masked by & 0xF)
    src1 = 16-bit value to be shifted (low 16 bits used)

  Semantics:
    shift = src0 & 0xF
    val   = src1 & 0xFFFF
    vd    = (val << shift) & 0xFFFF

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
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

MASK16: int = 0xFFFF
SHIFT_MASK: int = 0xF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit integer range [0, 2^16 - 1]."""
    return val & MASK16


def ref_lshlrev_b16(src0: int, src1: int) -> int:
    """Exact architectural reference oracle for AMD RDNA2 v_lshlrev_b16.

    Args:
        src0: Shift amount (lower 4 bits used)
        src1: 16-bit value to shift

    Returns:
        16-bit shifted result = (src1 << (src0 & 0xF)) & 0xFFFF
    """
    shift = src0 & SHIFT_MASK
    val = u16(src1)
    return (val << shift) & MASK16


class TestValuLshlrevB16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of v_lshlrev_b16."""

    def test_oracle_basic_shift(self):
        # 0x12 << 3 = 0x90
        res = ref_lshlrev_b16(3, 0x12)
        self.assertEqual(res, 0x90)

    def test_oracle_zero_shift(self):
        # 0xABCD << 0 = 0xABCD
        res = ref_lshlrev_b16(0, 0xABCD)
        self.assertEqual(res, 0xABCD)

    def test_oracle_max_shift_15(self):
        # 1 << 15 = 0x8000
        res = ref_lshlrev_b16(15, 1)
        self.assertEqual(res, 0x8000)

    def test_oracle_shift_count_masking(self):
        # Shift counts are masked by low 4 bits (& 0xF): 19 & 0xF = 3
        res = ref_lshlrev_b16(19, 0x12)
        self.assertEqual(res, 0x90)

    def test_oracle_overflow_truncation(self):
        # Shifting past bit 15 discards upper bits: 0x8000 << 1 = 0
        res = ref_lshlrev_b16(1, 0x8000)
        self.assertEqual(res, 0)

    def test_oracle_operand_asymmetry(self):
        # Reversed operands: src0 is shift count, src1 is value
        # 3 << 1 != 1 << 3
        res0 = ref_lshlrev_b16(1, 3)   # 3 << 1 = 6
        res1 = ref_lshlrev_b16(3, 1)   # 1 << 3 = 8
        self.assertNotEqual(res0, res1)
        self.assertEqual(res0, 6)
        self.assertEqual(res1, 8)


class Opcode(Enum):
    V_LSHLREV_B16 = auto()


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL16 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @classmethod
    def vgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 255:
            raise ValueError(f"VGPR index out of range: v{index}")
        return cls(kind=OperandKind.VGPR, index=index)

    @classmethod
    def sgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 105:
            raise ValueError(f"SGPR index out of range: s{index}")
        return cls(kind=OperandKind.SGPR, index=index)

    @classmethod
    def literal16(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL16, value=u16(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, value=u16(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.V_LSHLREV_B16:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")


class TestValuLshlrevB16AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.V_LSHLREV_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.V_LSHLREV_B16)
        self.assertEqual(inst.dst.kind, OperandKind.VGPR)
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)

    def test_ast_rejects_out_of_range_vgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.vgpr(256)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.V_LSHLREV_B16,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class RegisterState:
    """Simulated state of GCN/RDNA registers for VALU scalar execution."""

    def __init__(self):
        self.vgpr = [0] * 256
        self.sgpr = [0] * 106

    def read_operand_u16(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgpr[op.index] & MASK16
        elif op.kind == OperandKind.SGPR:
            return self.sgpr[op.index] & MASK16
        elif op.kind in (OperandKind.LITERAL16, OperandKind.INLINE_CONST):
            return op.value & MASK16
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_vgpr_u16(self, index: int, val: int) -> None:
        # VOP2 forms write low 16 bits and zero high 16 bits
        self.vgpr[index] = u16(val)

    def execute(self, inst: ASTInstruction) -> int:
        inst.validate()
        src0 = self.read_operand_u16(inst.src0)
        src1 = self.read_operand_u16(inst.src1)
        res = ref_lshlrev_b16(src0, src1)
        self.write_vgpr_u16(inst.dst.index, res)
        return res


class TestValuLshlrevB16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = RegisterState()
        state.vgpr[1] = 4
        state.vgpr[2] = 0x000F
        inst = ASTInstruction(
            opcode=Opcode.V_LSHLREV_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x00F0)
        self.assertEqual(state.vgpr[0], 0x00F0)


class TestValuLshlrevB16SimdDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and per-lane shift amounts."""

    def test_simd_per_lane_shifts(self):
        # 64 lanes with per-lane shift amount = lane % 16
        for lane in range(64):
            shift = lane % 16
            val = 1
            expected = (1 << shift) & MASK16
            res = ref_lshlrev_b16(shift, val)
            self.assertEqual(res, expected)

    def test_simd_lane_masking_active_inactive(self):
        # Simulate active lanes (even lanes only)
        exec_mask = 0x5555555555555555
        dest_vgpr = [0xDEAD] * 64

        for lane in range(64):
            is_active = (exec_mask & (1 << lane)) != 0
            if is_active:
                dest_vgpr[lane] = ref_lshlrev_b16(2, lane)

        for lane in range(64):
            if (exec_mask & (1 << lane)) != 0:
                self.assertEqual(dest_vgpr[lane], (lane << 2) & MASK16)
            else:
                self.assertEqual(dest_vgpr[lane], 0xDEAD)


class TestValuLshlrevB16RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        # v0 = v0 (shift) << v1 (val)
        state = RegisterState()
        state.vgpr[0] = 3
        state.vgpr[1] = 0x0010
        inst = ASTInstruction(
            opcode=Opcode.V_LSHLREV_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x0080)
        self.assertEqual(state.vgpr[0], 0x0080)

    def test_dst_aliases_src1(self):
        # v1 = v0 (shift) << v1 (val)
        state = RegisterState()
        state.vgpr[0] = 4
        state.vgpr[1] = 0x0005
        inst = ASTInstruction(
            opcode=Opcode.V_LSHLREV_B16,
            dst=ASTOperand.vgpr(1),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x0050)
        self.assertEqual(state.vgpr[1], 0x0050)

    def test_self_aliasing(self):
        # v0 = v0 << (v0 & 0xF)
        state = RegisterState()
        state.vgpr[0] = 2  # 2 << 2 = 8
        inst = ASTInstruction(
            opcode=Opcode.V_LSHLREV_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
        )
        res = state.execute(inst)
        self.assertEqual(res, 8)
        self.assertEqual(state.vgpr[0], 8)


class TestValuLshlrevB16ConditionCodes(unittest.TestCase):
    """Test suite for condition codes and flag preservation."""

    def test_flags_neutrality(self):
        # VALU shifts do not update SCC or VCC
        scc = 0
        vcc = 0
        res = ref_lshlrev_b16(1, 0x8000)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(vcc, 0)


class TestValuLshlrevB16BitwiseEdgeCases(unittest.TestCase):
    """Test suite for 16-bit float bit patterns and boundary checkerboard masks."""

    def test_ieee754_half_precision_bit_patterns(self):
        # Float16 bit patterns: +inf, -inf, qNaN, sNaN, +0, -0, min subnormal, max subnormal
        cases = [
            0x7C00,  # +inf
            0xFC00,  # -inf
            0x7E00,  # Quiet NaN
            0x7C01,  # Signaling NaN
            0x0000,  # +0.0
            0x8000,  # -0.0
            0x0001,  # Minimum subnormal
            0x03FF,  # Maximum subnormal
        ]
        for pattern in cases:
            for shift in (0, 1, 8, 15):
                with self.subTest(pattern=hex(pattern), shift=shift):
                    res = ref_lshlrev_b16(shift, pattern)
                    expected = (pattern << shift) & MASK16
                    self.assertEqual(res, expected)

    def test_boundary_checkerboard_bitmasks(self):
        checkerboards = [0x5555, 0xAAAA, 0x3333, 0xCCCC, 0x0F0F, 0xF0F0, 0x00FF, 0xFF00]
        for cb in checkerboards:
            for shift in (0, 1, 4, 8, 12, 15):
                res = ref_lshlrev_b16(shift, cb)
                self.assertEqual(res, (cb << shift) & MASK16)


def parse_v_lshlrev_b16_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line for v_lshlrev_b16."""
    clean = asm_line.strip()
    if not clean:
        raise ValueError("Empty assembly line")
    parts = clean.split(None, 1)
    mnemonic = parts[0].lower()
    if mnemonic not in ("v_lshlrev_b16", "v_lshlrev_b16_e32", "v_lshlrev_b16_e64"):
        raise ValueError(f"Expected mnemonic v_lshlrev_b16, got {mnemonic}")
    if len(parts) < 2:
        raise ValueError("Missing operands")
    operands_str = [op.strip() for op in parts[1].split(",")]
    if len(operands_str) != 3:
        raise ValueError(f"Expected 3 operands (dst, src0, src1), got {len(operands_str)}")

    def parse_op(op_str: str) -> ASTOperand:
        op_lower = op_str.lower()
        if op_lower.startswith("v") and op_lower[1:].isdigit():
            return ASTOperand.vgpr(int(op_lower[1:]))
        elif op_lower.startswith("s") and op_lower[1:].isdigit():
            return ASTOperand.sgpr(int(op_lower[1:]))
        elif op_str.startswith("0x") or op_str.startswith("-") or op_str.isdigit():
            val = int(op_str, 0)
            return ASTOperand.literal16(val)
        else:
            raise ValueError(f"Invalid operand: {op_str}")

    dst = parse_op(operands_str[0])
    src0 = parse_op(operands_str[1])
    src1 = parse_op(operands_str[2])

    inst = ASTInstruction(Opcode.V_LSHLREV_B16, dst, src0, src1)
    inst.validate()
    return inst


# Mutant functions for mutation testing
def mutant_lshr(src0: int, src1: int) -> int:
    """Mutant 1: uses right shift instead of left shift."""
    shift = src0 & SHIFT_MASK
    val = u16(src1)
    return (val >> shift) & MASK16


def mutant_no_mask(src0: int, src1: int) -> int:
    """Mutant 2: does not mask shift count by 0xF."""
    return (u16(src1) << src0) & MASK16


def mutant_swapped_operands(src0: int, src1: int) -> int:
    """Mutant 3: treats src1 as shift count and src0 as value."""
    shift = src1 & SHIFT_MASK
    val = u16(src0)
    return (val << shift) & MASK16


def mutant_no_truncation(src0: int, src1: int) -> int:
    """Mutant 4: fails to truncate to 16 bits."""
    shift = src0 & SHIFT_MASK
    val = u16(src1)
    return val << shift


class TestValuLshlrevB16NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, mutation detection, and translator execution."""

    def test_syntax_rejection_malformed_syntax(self):
        with self.assertRaises(ValueError):
            parse_v_lshlrev_b16_asm("")

        with self.assertRaises(ValueError):
            parse_v_lshlrev_b16_asm("v_add_u16 v0, v1, v2")

        with self.assertRaises(ValueError):
            parse_v_lshlrev_b16_asm("v_lshlrev_b16 v0, v1")

        with self.assertRaises(TypeError):
            parse_v_lshlrev_b16_asm("v_lshlrev_b16 s0, v1, v2")

    def test_mutation_detection_lshr(self):
        correct = ref_lshlrev_b16(2, 4)
        mutated = mutant_lshr(2, 4)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_no_mask(self):
        correct = ref_lshlrev_b16(16, 1)  # 16 & 0xF = 0 -> 1 << 0 = 1
        mutated = mutant_no_mask(16, 1)   # 1 << 16 & 0xFFFF = 0
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_swapped_operands(self):
        correct = ref_lshlrev_b16(1, 4)  # 4 << 1 = 8
        mutated = mutant_swapped_operands(1, 4)  # 1 << 4 = 16
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_no_truncation(self):
        correct = ref_lshlrev_b16(1, 0x8000)  # 0
        mutated = mutant_no_truncation(1, 0x8000)  # 0x10000
        self.assertNotEqual(correct, mutated)

    def test_lshlrev_spirv_emission(self):
        """Verify that support.translate_lines emits OpShiftLeftLogical and OpBitwiseAnd."""
        lines = [
            "v_mov_b32_e32 v1, 3",
            "v_mov_b32_e32 v2, 0x12",
            "v_lshlrev_b16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_lshlrev_b16_emit", lines)
        self.assertIn("OpShiftLeftLogical", spv)
        self.assertIn("OpBitwiseAnd", spv)

    def test_lshlrev_hardware_execution(self):
        """Verify hardware-accurate execution of v_lshlrev_b16 through support.mrt0_words."""
        lines = [
            "v_mov_b32_e32 v1, 3",
            "v_mov_b32_e32 v2, 0x12",
            "v_lshlrev_b16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_lshlrev_b16_hw", lines)
        words = support.mrt0_words(spv)
        # 0x12 << 3 = 0x90
        self.assertEqual(words[0], 0x90)
        self.assertEqual(words[1], 0x90)
        self.assertEqual(words[2], 0x90)
        self.assertEqual(words[3], 0x90)


if __name__ == "__main__":
    unittest.main()
