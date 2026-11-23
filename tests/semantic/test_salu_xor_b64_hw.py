from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_xor_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_xor_b64 instruction belongs to the SOP2 format.
  Instruction:
    s_xor_b64 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 64-bit destination SGPR pair s[2k:2k+1] (or EXEC / VCC)
    ssrc0 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC
    ssrc1 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC

  Semantics:
    sdst = (ssrc0 ^ ssrc1) & 0xFFFFFFFFFFFFFFFF
    SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
100% of test cases exercise tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words
for hardware-accurate translation and SPIR-V execution verification.
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
MASK64: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_ACTIVE: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_INACTIVE: int = 0x0000000000000000

_counter = 0


def next_test_id(prefix: str = "ps_xor64") -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer range [0, 2^64 - 1]."""
    return val & MASK64


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_xor_b64(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_xor_b64 instruction.

    Args:
        ssrc0: 64-bit first source operand (uint64)
        ssrc1: 64-bit second source operand (uint64)

    Returns:
        tuple (sdst, scc):
          - sdst: 64-bit result = (ssrc0 ^ ssrc1) & MASK64
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u64(ssrc0)
    s1 = u64(ssrc1)
    sdst = u64(s0 ^ s1)
    scc = 1 if sdst != 0 else 0
    return sdst, scc


def run_xor_b64(
    s0: int,
    s1: int,
    test_name: str | None = None,
    aliasing: str | None = None,
    scc_in: int | None = None,
) -> tuple[int, int]:
    """Execute s_xor_b64 end-to-end via support.translate_lines and support.mrt0_words.

    Args:
        s0: 64-bit first operand
        s1: 64-bit second operand
        test_name: optional unique shader name
        aliasing: None, 'dst_src0', 'dst_src1', 'all'
        scc_in: optional initial SCC value (0 or 1)

    Returns:
        (result_u64, scc)
    """
    name = test_name or next_test_id("ps_xor64_run")
    s0_lo = s0 & MASK32
    s0_hi = (s0 >> 32) & MASK32
    s1_lo = s1 & MASK32
    s1_hi = (s1 >> 32) & MASK32

    lines = []
    if scc_in is not None:
        if scc_in == 1:
            lines.append("s_cmp_eq_u32 0, 0")  # sets SCC = 1
        else:
            lines.append("s_cmp_lg_u32 0, 0")  # clears SCC = 0

    lines.extend([
        f"s_mov_b32 s0, {s0_lo:#x}",
        f"s_mov_b32 s1, {s0_hi:#x}",
        f"s_mov_b32 s2, {s1_lo:#x}",
        f"s_mov_b32 s3, {s1_hi:#x}",
    ])

    if aliasing == "dst_src0":
        lines.append("s_xor_b64 s[0:1], s[0:1], s[2:3]")
        dst_lo, dst_hi = "s0", "s1"
    elif aliasing == "dst_src1":
        lines.append("s_xor_b64 s[2:3], s[0:1], s[2:3]")
        dst_lo, dst_hi = "s2", "s3"
    elif aliasing == "all":
        lines.append("s_xor_b64 s[0:1], s[0:1], s[0:1]")
        dst_lo, dst_hi = "s0", "s1"
    else:
        lines.append("s_xor_b64 s[4:5], s[0:1], s[2:3]")
        dst_lo, dst_hi = "s4", "s5"

    lines.extend([
        "s_cselect_b32 s6, 1, 0",
        f"v_mov_b32_e32 v0, {dst_lo}",
        f"v_mov_b32_e32 v1, {dst_hi}",
        "v_mov_b32_e32 v2, s6",
        "v_mov_b32_e32 v3, 0",
        "exp mrt0, v0, v1, v2, v3 done",
        "s_endpgm",
    ])

    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    res_lo = words[0] & MASK32
    res_hi = words[1] & MASK32
    scc = words[2] & 1
    return (res_lo | (res_hi << 32)), scc


class TestSaluXorB64Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle verified via translator-backed hardware execution."""

    def test_oracle_basic_translation(self) -> None:
        s0 = 0x123456789ABCDEF0
        s1 = 0x00FF00FF0F0F0F0F
        ref_dst, ref_scc = ref_xor_b64(s0, s1)
        hw_dst, hw_scc = run_xor_b64(s0, s1, next_test_id("ps_oracle_basic"))
        self.assertEqual(hw_dst, ref_dst)
        self.assertEqual(hw_scc, ref_scc)
        self.assertEqual(hw_scc, 1)

    def test_oracle_self_xor_cancels_to_zero(self) -> None:
        val = 0xFEDCBA9876543210
        ref_dst, ref_scc = ref_xor_b64(val, val)
        hw_dst, hw_scc = run_xor_b64(val, val, next_test_id("ps_oracle_cancel"))
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)
        self.assertEqual(hw_dst, ref_dst)
        self.assertEqual(hw_scc, ref_scc)

    def test_oracle_zero_identity(self) -> None:
        val = 0x0123456789ABCDEF
        ref_dst, ref_scc = ref_xor_b64(val, 0)
        hw_dst, hw_scc = run_xor_b64(val, 0, next_test_id("ps_oracle_ident"))
        self.assertEqual(hw_dst, val)
        self.assertEqual(hw_scc, 1)
        self.assertEqual(hw_dst, ref_dst)
        self.assertEqual(hw_scc, ref_scc)

    def test_oracle_invert_with_all_ones(self) -> None:
        val = 0x5555555555555555
        ref_dst, ref_scc = ref_xor_b64(val, ALL_LANES_ACTIVE)
        hw_dst, hw_scc = run_xor_b64(val, ALL_LANES_ACTIVE, next_test_id("ps_oracle_invert"))
        self.assertEqual(hw_dst, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(hw_scc, 1)
        self.assertEqual(hw_dst, ref_dst)
        self.assertEqual(hw_scc, ref_scc)

    def test_oracle_commutativity(self) -> None:
        s0 = 0x123456789ABCDEF0
        s1 = 0xFEDCBA9876543210
        hw01, scc01 = run_xor_b64(s0, s1, next_test_id("ps_comm_01"))
        hw10, scc10 = run_xor_b64(s1, s0, next_test_id("ps_comm_10"))
        self.assertEqual(hw01, hw10)
        self.assertEqual(scc01, scc10)

    def test_oracle_alternating_bitmasks(self) -> None:
        s0 = 0x3333333333333333
        s1 = 0xCCCCCCCCCCCCCCCC
        hw_dst, hw_scc = run_xor_b64(s0, s1, next_test_id("ps_oracle_alt"))
        self.assertEqual(hw_dst, ALL_LANES_ACTIVE)
        self.assertEqual(hw_scc, 1)


class Opcode(Enum):
    S_XOR_B64 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    LITERAL64 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 2
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 2) -> ASTOperand:
        if count != 2:
            raise ValueError(f"s_xor_b64 requires 64-bit register pair (count=2), got {count}")
        if index < 0 or index > 104 or (index + count) > 106:
            raise ValueError(f"SGPR index out of range: s[{index}:{index + count - 1}]")
        if index % 2 != 0:
            raise ValueError(f"SGPR pair must be 64-bit aligned (even index), got s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_mask(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC, count=2)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC, count=2)

    @classmethod
    def literal64(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL64, count=2, value=u64(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=2, value=u64(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.SGPR:
                return f"s[{op.index}:{op.index + 1}]"
            if op.kind == OperandKind.EXEC:
                return "exec"
            if op.kind == OperandKind.VCC:
                return "vcc"
            return hex(op.value)
        return f"s_xor_b64 {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}"


class TestSaluXorB64AST(unittest.TestCase):
    """Stage 2: AST opcode definitions, typed expressions, and translator emission."""

    def test_ast_construction_sgpr_pair(self) -> None:
        dst = ASTOperand.sgpr(0)
        src0 = ASTOperand.sgpr(2)
        src1 = ASTOperand.sgpr(4)
        inst = ASTInstruction(Opcode.S_XOR_B64, dst, src0, src1)
        self.assertEqual(inst.disasm(), "s_xor_b64 s[0:1], s[2:3], s[4:5]")

    def test_ast_odd_alignment_rejection(self) -> None:
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(3)

    def test_ast_out_of_range_rejection(self) -> None:
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(106)

    def test_ast_translator_emission(self) -> None:
        dst = ASTOperand.sgpr(4)
        src0 = ASTOperand.sgpr(0)
        src1 = ASTOperand.sgpr(2)
        inst = ASTInstruction(Opcode.S_XOR_B64, dst, src0, src1)
        lines = [
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_mov_b32 s2, 0x33333333",
            "s_mov_b32 s3, 0x44444444",
            inst.disasm(),
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_ast_emit"), lines)
        self.assertIn("OpBitwiseXor", spv)


class TestSaluXorB64ScalarExecution(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction execution through translator backend."""

    def test_scalar_execution_arbitrary_pattern(self) -> None:
        s0 = 0x00000000FFFFFFFF
        s1 = 0x00000000FFFF0000
        hw_dst, hw_scc = run_xor_b64(s0, s1, next_test_id("ps_scal_arb"))
        self.assertEqual(hw_dst, 0x000000000000FFFF)
        self.assertEqual(hw_scc, 1)

    def test_scalar_execution_high_dword_only(self) -> None:
        s0 = 0xFFFFFFFF00000000
        s1 = 0x0000000000000000
        hw_dst, hw_scc = run_xor_b64(s0, s1, next_test_id("ps_scal_hi"))
        self.assertEqual(hw_dst, 0xFFFFFFFF00000000)
        self.assertEqual(hw_scc, 1)

    def test_scalar_execution_low_dword_only(self) -> None:
        s0 = 0x0000000000000001
        s1 = 0x0000000000000000
        hw_dst, hw_scc = run_xor_b64(s0, s1, next_test_id("ps_scal_lo"))
        self.assertEqual(hw_dst, 0x0000000000000001)
        self.assertEqual(hw_scc, 1)

    def test_scalar_execution_unaffected_registers(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_mov_b32 s2, 0x00ff00ff",
            "s_mov_b32 s3, 0x0f0f0f0f",
            "s_mov_b32 s6, 0xCAFEBABE",
            "s_mov_b32 s7, 0xDEADBEEF",
            "s_xor_b64 s[4:5], s[0:1], s[2:3]",
            "v_mov_b32_e32 v0, s6",
            "v_mov_b32_e32 v1, s7",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_unaffected"), lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xCAFEBABE)
        self.assertEqual(words[1], 0xDEADBEEF)


class TestSaluXorB64SIMDAndMasking(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 64-bit EXEC/VCC mask interaction."""

    def test_exec_mask_xor_emission(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x55555555",
            "s_mov_b32 s1, 0x55555555",
            "s_xor_b64 exec, exec, s[0:1]",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_exec_emit"), lines)
        self.assertIn("OpBitwiseXor", spv)

    def test_vcc_mask_xor_emission(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x33333333",
            "s_mov_b32 s1, 0x33333333",
            "s_xor_b64 vcc, vcc, s[0:1]",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_vcc_emit"), lines)
        self.assertIn("OpBitwiseXor", spv)

    def test_inline_literal_invert(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_xor_b64 s[4:5], s[0:1], -1",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_lit_neg1"), lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345678 ^ 0xFFFFFFFF)
        self.assertEqual(words[1], 0x9ABCDEF0 ^ 0xFFFFFFFF)


class TestSaluXorB64Aliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation."""

    def test_aliasing_dst_src0(self) -> None:
        s0 = 0xF0F0F0F0F0F0F0F0
        s1 = 0x0F0F0F0F0F0F0F0F
        hw_dst, hw_scc = run_xor_b64(s0, s1, next_test_id("ps_alias_dst_s0"), aliasing="dst_src0")
        self.assertEqual(hw_dst, ALL_LANES_ACTIVE)
        self.assertEqual(hw_scc, 1)

    def test_aliasing_dst_src1(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_mov_b32 s2, 0x00ff00ff",
            "s_mov_b32 s3, 0x0f0f0f0f",
            "s_xor_b64 s[2:3], s[0:1], s[2:3]",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s1",
            "v_mov_b32_e32 v2, s2",
            "v_mov_b32_e32 v3, s3",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_alias_dst_s1"), lines)
        words = support.mrt0_words(spv)
        # s[0:1] preserved unchanged
        self.assertEqual(words[0], 0x12345678)
        self.assertEqual(words[1], 0x9ABCDEF0)
        # s[2:3] holds XOR result
        self.assertEqual(words[2], 0x12345678 ^ 0x00FF00FF)
        self.assertEqual(words[3], 0x9ABCDEF0 ^ 0x0F0F0F0F)

    def test_aliasing_self_cancel(self) -> None:
        s0 = 0x123456789ABCDEF0
        hw_dst, hw_scc = run_xor_b64(s0, s0, next_test_id("ps_alias_cancel"), aliasing="all")
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_aliasing_chained_invert_restore(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_mov_b32 s2, 0x55555555",
            "s_mov_b32 s3, 0x55555555",
            "s_xor_b64 s[0:1], s[0:1], s[2:3]",
            "s_xor_b64 s[0:1], s[0:1], s[2:3]",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s1",
            "exp mrt0, v0, v1, v0, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_alias_chain"), lines)
        words = support.mrt0_words(spv)
        # (A ^ B) ^ B == A exact restoration
        self.assertEqual(words[0], 0x12345678)
        self.assertEqual(words[1], 0x9ABCDEF0)


class TestSaluXorB64ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and SCC generation."""

    def test_scc_set_when_nonzero(self) -> None:
        hw_dst, hw_scc = run_xor_b64(1, 0, next_test_id("ps_scc_nz"))
        self.assertEqual(hw_dst, 1)
        self.assertEqual(hw_scc, 1)

    def test_scc_cleared_when_zero(self) -> None:
        hw_dst, hw_scc = run_xor_b64(
            0x123456789ABCDEF0,
            0x123456789ABCDEF0,
            next_test_id("ps_scc_zero"),
            scc_in=1,
        )
        self.assertEqual(hw_dst, 0)
        self.assertEqual(hw_scc, 0)

    def test_scc_set_when_only_lo_nonzero(self) -> None:
        hw_dst, hw_scc = run_xor_b64(0x0000000080000000, 0, next_test_id("ps_scc_lo"))
        self.assertEqual(hw_dst, 0x0000000080000000)
        self.assertEqual(hw_scc, 1)

    def test_scc_set_when_only_hi_nonzero(self) -> None:
        hw_dst, hw_scc = run_xor_b64(0x8000000000000000, 0, next_test_id("ps_scc_hi"))
        self.assertEqual(hw_dst, 0x8000000000000000)
        self.assertEqual(hw_scc, 1)

    def test_scc_previous_scc_overwritten(self) -> None:
        # Initial SCC=0, but result is nonzero -> SCC becomes 1
        _, scc_to_1 = run_xor_b64(0x10, 0x00, next_test_id("ps_scc_to1"), scc_in=0)
        self.assertEqual(scc_to_1, 1)

        # Initial SCC=1, but result is zero -> SCC becomes 0
        _, scc_to_0 = run_xor_b64(0x10, 0x10, next_test_id("ps_scc_to0"), scc_in=1)
        self.assertEqual(scc_to_0, 0)


class TestSaluXorB64IEEE754(unittest.TestCase):
    """Stage 7: IEEE 754 boundary extremes, subnormals, and NaN bit-pattern handling."""

    def test_float_pattern_preservation(self) -> None:
        pos_zero = 0x0000000000000000
        neg_zero = 0x8000000000000000
        hw_dst, hw_scc = run_xor_b64(pos_zero, neg_zero, next_test_id("ps_ieee_zeros"))
        self.assertEqual(hw_dst, neg_zero)
        self.assertEqual(hw_scc, 1)

    def test_nan_payload_xor(self) -> None:
        qnan64 = 0x7FF8000000000000
        snan64 = 0x7FF0000000000001
        ref_dst, ref_scc = ref_xor_b64(qnan64, snan64)
        hw_dst, hw_scc = run_xor_b64(qnan64, snan64, next_test_id("ps_ieee_nan"))
        self.assertEqual(hw_dst, ref_dst)
        self.assertEqual(hw_scc, ref_scc)

    def test_inf_pattern_xor(self) -> None:
        pos_inf = 0x7FF0000000000000
        neg_inf = 0xFFF0000000000000
        hw_dst, hw_scc = run_xor_b64(pos_inf, neg_inf, next_test_id("ps_ieee_inf"))
        self.assertEqual(hw_dst, 0x8000000000000000)
        self.assertEqual(hw_scc, 1)


class TestSaluXorB64NegativeControls(unittest.TestCase):
    """Stage 8: Negative controls, syntax rejection, and mutation detection."""

    def test_mutation_or_instead_of_xor(self) -> None:
        def mutant_or(s0: int, s1: int) -> int:
            return s0 | s1
        correct, _ = run_xor_b64(0x0F, 0x0F, next_test_id("ps_mut_or"))
        mutated = mutant_or(0x0F, 0x0F)
        self.assertNotEqual(correct, mutated)

    def test_mutation_and_instead_of_xor(self) -> None:
        def mutant_and(s0: int, s1: int) -> int:
            return s0 & s1
        correct, _ = run_xor_b64(0x0F, 0xF0, next_test_id("ps_mut_and"))
        mutated = mutant_and(0x0F, 0xF0)
        self.assertNotEqual(correct, mutated)

    def test_mutation_drop_hi_dword(self) -> None:
        def mutant_drop_hi(s0: int, s1: int) -> int:
            return (s0 ^ s1) & MASK32
        val0 = 0x1234567800000000
        val1 = 0x0000000000000000
        correct, _ = run_xor_b64(val0, val1, next_test_id("ps_mut_hi"))
        mutated = mutant_drop_hi(val0, val1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_invert_scc(self) -> None:
        def mutant_inv_scc(res: int) -> int:
            return 1 if res == 0 else 0
        _, correct_scc = run_xor_b64(0, 0, next_test_id("ps_mut_scc"))
        mutated_scc = mutant_inv_scc(0)
        self.assertNotEqual(correct_scc, mutated_scc)

    def test_spirv_emission_contains_xor(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_mov_b32 s2, 0x00ff00ff",
            "s_mov_b32 s3, 0x0f0f0f0f",
            "s_xor_b64 s[4:5], s[0:1], s[2:3]",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_neg_emit"), lines)
        self.assertIn("OpBitwiseXor", spv)


if __name__ == "__main__":
    unittest.main()
