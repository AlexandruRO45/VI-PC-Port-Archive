from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_andn2_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_andn2_b32 instruction belongs to SOP2 format.
  Instruction:
    s_andn2_b32 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo)
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo
    ssrc1 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = ssrc0 & ~ssrc1
    SCC = 1 if sdst != 0 else 0

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

MASK32: int = 0xFFFFFFFF
ALL_LANES_ACTIVE_32: int = 0xFFFFFFFF
ALL_LANES_INACTIVE_32: int = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_andn2_b32(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_andn2_b32 instruction.

    Args:
        ssrc0: 32-bit first source operand (uint32)
        ssrc1: 32-bit second source operand (uint32)

    Returns:
        tuple (sdst, scc):
          - sdst: 32-bit result = ssrc0 & ~ssrc1
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u32(ssrc0)
    s1 = u32(ssrc1)
    sdst = u32(s0 & (~s1 & MASK32))
    scc = 1 if sdst != 0 else 0
    return sdst, scc


def run_hw_andn2(ssrc0: int, ssrc1: int, name: str = "ps_andn2_run") -> tuple[int, int]:
    """Execute s_andn2_b32 through rdna2spirv translator and return (sdst, scc)."""
    s0 = u32(ssrc0)
    s1 = u32(ssrc1)
    lines = [
        f"s_mov_b32 s0, {s0:#x}",
        f"s_mov_b32 s1, {s1:#x}",
        "s_andn2_b32 s2, s0, s1",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32_e32 v0, s2",
        "v_mov_b32_e32 v1, s3",
        "exp mrt0, v0, v1, v0, v1 done",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class TestSaluAndn2B32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_andn2_b32."""

    def test_oracle_basic(self):
        s0 = 0x12345678
        s1 = 0x00FF00FF
        sdst, scc = ref_andn2_b32(s0, s1)
        expected = s0 & (~s1 & MASK32)
        self.assertEqual(sdst, expected)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(s0, s1, "ps_oracle_basic")
        self.assertEqual(hw_dst, expected)
        self.assertEqual(hw_scc, 1)

    def test_oracle_zero_s0(self):
        s1 = 0x12345678
        sdst, scc = ref_andn2_b32(0, s1)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(0, s1, "ps_oracle_zero_s0")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_oracle_all_ones_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(s0, ALL_LANES_ACTIVE_32, "ps_oracle_all_ones_s1")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_oracle_zero_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, 0)
        self.assertEqual(sdst, s0)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(s0, 0, "ps_oracle_zero_s1")
        self.assertEqual(hw_dst, s0)
        self.assertEqual(hw_scc, 1)

    def test_oracle_operand_asymmetry(self):
        # s0 & ~s1 != s1 & ~s0 (non-commutative)
        s0 = 0x0000FFFF
        s1 = 0x00FF00FF
        res_01, _ = ref_andn2_b32(s0, s1)
        res_10, _ = ref_andn2_b32(s1, s0)
        self.assertNotEqual(res_01, res_10)
        self.assertEqual(res_01, 0x0000FF00)
        self.assertEqual(res_10, 0x00FF0000)
        hw_01, _ = run_hw_andn2(s0, s1, "ps_oracle_asym_01")
        hw_10, _ = run_hw_andn2(s1, s0, "ps_oracle_asym_10")
        self.assertNotEqual(hw_01, hw_10)
        self.assertEqual(hw_01, 0x0000FF00)
        self.assertEqual(hw_10, 0x00FF0000)


class Opcode(Enum):
    S_ANDN2_B32 = auto()


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
    def sgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 105:
            raise ValueError(f"SGPR index out of range: s{index}")
        return cls(kind=OperandKind.SGPR, index=index)

    @classmethod
    def exec_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC_LO)

    @classmethod
    def vcc_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC_LO)

    @classmethod
    def literal32(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL32, value=u32(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, value=u32(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.S_ANDN2_B32:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC_LO, OperandKind.VCC_LO):
            raise TypeError(f"Destination must be SGPR, EXEC_LO, or VCC_LO, got {self.dst.kind}")
        if self.dst.count != 1 or self.src0.count != 1 or self.src1.count != 1:
            raise ValueError("All operands must be 32-bit single registers")


class TestSaluAndn2B32AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.sgpr(4),
            src0=ASTOperand.sgpr(6),
            src1=ASTOperand.sgpr(8),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_ANDN2_B32)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)
        self.assertEqual(inst.src1.index, 8)

    def test_ast_rejects_out_of_range_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(106)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.literal32(0x123),
            src0=ASTOperand.sgpr(6),
            src1=ASTOperand.sgpr(8),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class ScalarRegisterState:
    """Simulated state of GCN/RDNA Scalar Registers (SGPRs, EXEC_LO, and SCC)."""

    def __init__(self, exec_lo: int = ALL_LANES_ACTIVE_32):
        self.sgpr = [0] * 106
        self.exec_lo = u32(exec_lo)
        self.vcc_lo = 0
        self.scc = 0

    def read_operand_u32(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.SGPR:
            return self.sgpr[op.index]
        elif op.kind == OperandKind.EXEC_LO:
            return self.exec_lo
        elif op.kind == OperandKind.VCC_LO:
            return self.vcc_lo
        elif op.kind in (OperandKind.LITERAL32, OperandKind.INLINE_CONST):
            return op.value
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_sgpr_u32(self, index: int, val: int) -> None:
        self.sgpr[index] = u32(val)

    def execute(self, inst: ASTInstruction) -> tuple[int, int]:
        inst.validate()
        s0 = self.read_operand_u32(inst.src0)
        s1 = self.read_operand_u32(inst.src1)
        sdst_val, scc = ref_andn2_b32(s0, s1)

        if inst.dst.kind == OperandKind.SGPR:
            self.write_sgpr_u32(inst.dst.index, sdst_val)
        elif inst.dst.kind == OperandKind.EXEC_LO:
            self.exec_lo = sdst_val
        elif inst.dst.kind == OperandKind.VCC_LO:
            self.vcc_lo = sdst_val

        self.scc = scc
        return sdst_val, scc


class TestSaluAndn2B32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = ScalarRegisterState()
        state.write_sgpr_u32(0, 0x12345678)
        state.write_sgpr_u32(2, 0x00FF00FF)
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.sgpr(4),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.sgpr(2),
        )
        sdst, scc = state.execute(inst)
        expected = 0x12345678 & (~0x00FF00FF & MASK32)
        self.assertEqual(sdst, expected)
        self.assertEqual(state.sgpr[4], expected)
        self.assertEqual(state.scc, 1)
        hw_dst, hw_scc = run_hw_andn2(0x12345678, 0x00FF00FF, "ps_interp_crosscheck")
        self.assertEqual(hw_dst, expected)
        self.assertEqual(hw_scc, 1)


class TestSaluAndn2B32SimdDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32-bit EXEC masking."""

    def test_wave32_lane_mask_isolation(self):
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        s0 = lanes(0, 24)  # lanes 0..23
        s1 = lanes(8, 32)  # lanes 8..31
        # s0 & ~s1 = lanes 0..7
        sdst, scc = ref_andn2_b32(s0, s1)
        self.assertEqual(sdst, lanes(0, 8))
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(s0, s1, "ps_lane_mask_iso")
        self.assertEqual(hw_dst, lanes(0, 8))
        self.assertEqual(hw_scc, 1)

    def test_divergent_mask_checkerboard(self):
        even_lanes = 0x55555555
        odd_lanes = 0xAAAAAAAA
        # even & ~odd = even & even = even
        sdst, scc = ref_andn2_b32(even_lanes, odd_lanes)
        self.assertEqual(sdst, even_lanes)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(even_lanes, odd_lanes, "ps_chk_divergent")
        self.assertEqual(hw_dst, even_lanes)
        self.assertEqual(hw_scc, 1)

        # even & ~even = 0
        sdst_zero, scc_zero = ref_andn2_b32(even_lanes, even_lanes)
        self.assertEqual(sdst_zero, 0)
        self.assertEqual(scc_zero, 0)
        hw_zero, hw_scc_z = run_hw_andn2(even_lanes, even_lanes, "ps_chk_zero")
        self.assertEqual(hw_zero, 0)
        self.assertEqual(hw_scc_z, 0)


class TestSaluAndn2B32RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        # s0 = s0 & ~s2
        state = ScalarRegisterState()
        s0_val = 0x12345678
        s1_val = 0x000000FF
        state.write_sgpr_u32(0, s0_val)
        state.write_sgpr_u32(2, s1_val)

        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.sgpr(2),
        )
        sdst, scc = state.execute(inst)
        self.assertEqual(sdst, 0x12345600)
        self.assertEqual(state.sgpr[0], 0x12345600)
        self.assertEqual(scc, 1)

        lines = [
            f"s_mov_b32 s0, {s0_val:#x}",
            f"s_mov_b32 s2, {s1_val:#x}",
            "s_andn2_b32 s0, s0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345600)
        self.assertEqual(words[1], 1)

    def test_dst_aliases_src1(self):
        # s2 = s0 & ~s2
        state = ScalarRegisterState()
        s0_val = 0x12345678
        s1_val = 0x000000FF
        state.write_sgpr_u32(0, s0_val)
        state.write_sgpr_u32(2, s1_val)

        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.sgpr(2),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.sgpr(2),
        )
        sdst, scc = state.execute(inst)
        self.assertEqual(sdst, 0x12345600)
        self.assertEqual(state.sgpr[2], 0x12345600)
        self.assertEqual(scc, 1)

        lines = [
            f"s_mov_b32 s0, {s0_val:#x}",
            f"s_mov_b32 s2, {s1_val:#x}",
            "s_andn2_b32 s2, s0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src1", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345600)
        self.assertEqual(words[1], 1)

    def test_self_cancellation_aliasing(self):
        # s0 = s0 & ~s0 = 0
        state = ScalarRegisterState()
        state.write_sgpr_u32(0, 0x12345678)
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.sgpr(0),
        )
        sdst, scc = state.execute(inst)
        self.assertEqual(sdst, 0)
        self.assertEqual(state.sgpr[0], 0)
        self.assertEqual(scc, 0)

        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s0, s0, s0",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_self_cancel", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)


class TestSaluAndn2B32ConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and SCC generation."""

    def test_scc_set_when_result_nonzero(self):
        sdst, scc = ref_andn2_b32(1, 0)
        self.assertEqual(sdst, 1)
        self.assertEqual(scc, 1)
        hw_dst, hw_scc = run_hw_andn2(1, 0, "ps_scc_nonzero")
        self.assertEqual(hw_dst, 1)
        self.assertEqual(hw_scc, 1)

    def test_scc_cleared_when_result_zero(self):
        sdst, scc = ref_andn2_b32(0x1234, 0x1234)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(0x1234, 0x1234, "ps_scc_zero")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_scc_cleared_when_src0_is_zero(self):
        sdst, scc = ref_andn2_b32(0, 0x55555555)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)
        hw_dst, hw_scc = run_hw_andn2(0, 0x55555555, "ps_scc_zero_src0")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)


class TestSaluAndn2B32BitwiseEdgeCases(unittest.TestCase):
    """Test suite for 32-bit IEEE-754 bit representations and boundary bitmasks."""

    def test_ieee754_single_precision_bit_patterns(self):
        cases = [
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC00000,  # Quiet NaN
            0x7F800001,  # Signaling NaN
            0x00000000,  # +0.0
            0x80000000,  # -0.0
            0x00000001,  # Minimum subnormal
            0x007FFFFF,  # Maximum subnormal
        ]
        for p0 in cases:
            for p1 in cases:
                with self.subTest(p0=hex(p0), p1=hex(p1)):
                    sdst, scc = ref_andn2_b32(p0, p1)
                    expected = p0 & (~p1 & MASK32)
                    self.assertEqual(sdst, expected)
                    self.assertEqual(scc, int(expected != 0))

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
        for cb0 in checkerboards:
            for cb1 in checkerboards:
                sdst, scc = ref_andn2_b32(cb0, cb1)
                self.assertEqual(sdst, cb0 & (~cb1 & MASK32))
        hw_dst1, hw_scc1 = run_hw_andn2(0x55555555, 0xAAAAAAAA, "ps_cb_5_a")
        self.assertEqual(hw_dst1, 0x55555555)
        self.assertEqual(hw_scc1, 1)
        hw_dst2, hw_scc2 = run_hw_andn2(0x33333333, 0xCCCCCCCC, "ps_cb_3_c")
        self.assertEqual(hw_dst2, 0x33333333)
        self.assertEqual(hw_scc2, 1)
        hw_dst3, hw_scc3 = run_hw_andn2(0x0F0F0F0F, 0xF0F0F0F0, "ps_cb_0f_f0")
        self.assertEqual(hw_dst3, 0x0F0F0F0F)
        self.assertEqual(hw_scc3, 1)


def parse_s_andn2_b32_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line for s_andn2_b32."""
    clean = asm_line.strip()
    if not clean:
        raise ValueError("Empty assembly line")
    parts = clean.split(None, 1)
    mnemonic = parts[0].lower()
    if mnemonic != "s_andn2_b32":
        raise ValueError(f"Expected mnemonic s_andn2_b32, got {mnemonic}")
    if len(parts) < 2:
        raise ValueError("Missing operands")
    operands_str = [op.strip() for op in parts[1].split(",")]
    if len(operands_str) != 3:
        raise ValueError(f"Expected 3 operands (dst, src0, src1), got {len(operands_str)}")

    def parse_op(op_str: str) -> ASTOperand:
        op_lower = op_str.lower()
        if op_lower in ("exec_lo", "exec"):
            return ASTOperand.exec_lo()
        elif op_lower in ("vcc_lo", "vcc"):
            return ASTOperand.vcc_lo()
        elif op_lower.startswith("s") and op_lower[1:].isdigit():
            return ASTOperand.sgpr(int(op_lower[1:]))
        elif op_str.startswith("0x") or op_str.startswith("-") or op_str.isdigit():
            val = int(op_str, 0)
            return ASTOperand.literal32(val)
        else:
            raise ValueError(f"Invalid operand: {op_str}")

    dst = parse_op(operands_str[0])
    src0 = parse_op(operands_str[1])
    src1 = parse_op(operands_str[2])

    inst = ASTInstruction(Opcode.S_ANDN2_B32, dst, src0, src1)
    inst.validate()
    return inst


# Mutant functions for mutation testing
def mutant_and(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 1: uses regular AND (omits ~ negation)."""
    res = (s0 & s1) & MASK32
    return res, 1 if res != 0 else 0


def mutant_orn2(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 2: uses OR instead of AND."""
    res = (s0 | (~s1 & MASK32)) & MASK32
    return res, 1 if res != 0 else 0


def mutant_swapped_operands(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 3: swaps operand order (s1 & ~s0)."""
    res = (s1 & (~s0 & MASK32)) & MASK32
    return res, 1 if res != 0 else 0


def mutant_invert_scc(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 4: inverts SCC logic."""
    res = (s0 & (~s1 & MASK32)) & MASK32
    return res, 1 if res == 0 else 0


class TestSaluAndn2B32NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, mutation detection, and translator execution."""

    def test_syntax_rejection_malformed_syntax(self):
        with self.assertRaises(ValueError):
            parse_s_andn2_b32_asm("")

        with self.assertRaises(ValueError):
            parse_s_andn2_b32_asm("s_orn2_b32 s0, s1, s2")

        with self.assertRaises(ValueError):
            parse_s_andn2_b32_asm("s_andn2_b32 s0, s1")

        with self.assertRaises(ValueError):
            parse_s_andn2_b32_asm("s_andn2_b32 s106, s1, s2")

    def test_mutation_detection_and(self):
        s0, s1 = 0x0F, 0x0F
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_and(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_orn2(self):
        s0, s1 = 0x00, 0x00
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_orn2(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_swapped_operands(self):
        s0, s1 = 0x0F, 0x00
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_swapped_operands(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_invert_scc(self):
        s0, s1 = 0x00, 0x00
        correct = ref_andn2_b32(s0, s1)
        mutated = mutant_invert_scc(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_andn2_spirv_emission(self):
        """Verify that support.translate_lines emits expected SPIR-V logic for s_andn2_b32."""
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x00ff00ff",
            "s_andn2_b32 s2, s0, s1",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_andn2_b32_emit", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)

    def test_andn2_hardware_execution(self):
        """Verify hardware-accurate execution of s_andn2_b32 through support.mrt0_words."""
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x00ff00ff",
            "s_andn2_b32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_b32_hw", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12005600)
        self.assertEqual(words[1], 1)

    def test_andn2_spirv_lowering_operand_variants(self):
        """Verify that support.translate_lines lowers operand variants with OpBitwiseAnd and OpNot."""
        spv_lit0 = support.translate_lines("ps_andn2_lit0", [
            "s_mov_b32 s1, 0x00ff00ff",
            "s_andn2_b32 s2, 0x12345678, s1",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_lit0)
        self.assertIn("OpNot", spv_lit0)

        spv_lit1 = support.translate_lines("ps_andn2_lit1", [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s2, s0, 0x00ff00ff",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_lit1)
        self.assertIn("OpNot", spv_lit1)

        spv_zero = support.translate_lines("ps_andn2_zero", [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s2, s0, 0",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_zero)
        self.assertIn("OpNot", spv_zero)

        spv_exec = support.translate_lines("ps_andn2_exec", [
            "s_mov_b32 s0, 0",
            "s_andn2_b32 exec_lo, exec_lo, s0",
            "s_endpgm",
        ])
        self.assertIn("OpBitwiseAnd", spv_exec)
        self.assertIn("OpNot", spv_exec)

    def test_andn2_hardware_execution_inline_constants(self):
        """Verify execution of inline 0 and -1 constants via support.mrt0_words."""
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s1, s0, 0",
            "s_andn2_b32 s2, s0, -1",
            "v_mov_b32_e32 v0, s1",
            "v_mov_b32_e32 v1, s2",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_inline_consts", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345678)
        self.assertEqual(words[1], 0)

    def test_andn2_hardware_execution_literal_constants(self):
        """Verify execution with 32-bit literal constants via support.mrt0_words."""
        lines = [
            "s_mov_b32 s1, 0x00ff00ff",
            "s_andn2_b32 s2, 0x12345678, s1",
            "s_mov_b32 s0, 0x12345678",
            "s_andn2_b32 s3, s0, 0x0000ffff",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_literals", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12005600)
        self.assertEqual(words[1], 0x12340000)

    def test_andn2_hardware_execution_sign_bit_boundary(self):
        """Verify execution on sign-bit and highest 32-bit boundary patterns."""
        lines = [
            "s_mov_b32 s0, 0x80000000",
            "s_mov_b32 s1, 0x7fffffff",
            "s_andn2_b32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_sign_boundary", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x80000000)
        self.assertEqual(words[1], 1)

    def test_andn2_negative_control_mutated_operator_fails(self):
        """Verify that replacing OpBitwiseAnd with an incorrect logic op produces detectable failure."""
        correct_dst, _ = run_hw_andn2(0x12345678, 0x00FF00FF, "ps_neg_ctrl_sample")
        mutated_dst = 0x12345678 | (~0x00FF00FF & MASK32)
        self.assertNotEqual(correct_dst, mutated_dst)


if __name__ == "__main__":
    unittest.main()
