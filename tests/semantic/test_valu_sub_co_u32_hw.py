from __future__ import annotations

"""Architectural test suite for AMD RDNA2 v_sub_co_u32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
    v_sub_co_u32 D, SDST, S0, S1 [clamp]

Semantic operation:
    D.u32 = (S0.u32 - S1.u32) & 0xFFFFFFFF
    SDST  = 1 if (S0.u32 < S1.u32) else 0

Hardware semantics:
    - 32-bit unsigned vector subtraction with carry-out / borrow generation.
    - SDST receives borrow-out flag (vcc_lo, arbitrary SGPR, or null).
    - VOP3 supports clamp modifier: saturates to 0 on unsigned borrow/underflow,
      while SDST borrow flag still reflects the borrow condition.
    - Boundary extrema, sign-bit transitions, and multi-precision 64-bit borrow chain.
    - 100% translator-backed execution through tools/rdna2spirv.py and spirv_mini_interp.

Standard library Python only (unittest, struct, os, sys).
Strictly zero game data.
"""

import os
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv
import support

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

_TRANSLATE_ID = 0


def _run_sub_co_u32(src0: int, src1: int, clamp: bool = False, sdst: str = "vcc_lo") -> tuple[int, int]:
    """Execute v_sub_co_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    clamp_suffix = " clamp" if clamp else ""
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    if sdst == "vcc_lo":
        lines = [
            f"v_mov_b32 v1, {u0:#x}",
            f"v_mov_b32 v2, {u1:#x}",
            f"v_sub_co_u32 v0, vcc_lo, v1, v2{clamp_suffix}",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, 0, 0 done vm",
            "s_endpgm",
        ]
    elif sdst == "null":
        lines = [
            f"v_mov_b32 v1, {u0:#x}",
            f"v_mov_b32 v2, {u1:#x}",
            f"v_sub_co_u32 v0, null, v1, v2{clamp_suffix}",
            "exp mrt0 v0, 0, 0, 0 done vm",
            "s_endpgm",
        ]
    else:
        # SGPR destination
        lines = [
            f"s_mov_b32 {sdst}, 0",
            f"v_mov_b32 v1, {u0:#x}",
            f"v_mov_b32 v2, {u1:#x}",
            f"v_sub_co_u32 v0, {sdst}, v1, v2{clamp_suffix}",
            f"v_cndmask_b32 v3, 0, 1, {sdst}",
            "exp mrt0 v0, v3, 0, 0 done vm",
            "s_endpgm",
        ]
    spv = support.translate_lines(f"ps_sub_co_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32, words[1] & 1


class TestValuSubCoU32TranslatorExecution(unittest.TestCase):
    """Direct translator-backed execution of v_sub_co_u32 subtraction and borrow generation."""

    def test_basic_subtraction_no_borrow(self) -> None:
        cases = [
            (10, 3, 7, 0),
            (100, 100, 0, 0),
            (0, 0, 0, 0),
            (1000, 400, 600, 0),
            (0x12345678, 0x11111111, 0x01234567, 0),
        ]
        for a, b, exp_diff, exp_borrow in cases:
            diff, borrow = _run_sub_co_u32(a, b)
            self.assertEqual(diff, exp_diff, f"Mismatch diff for {a} - {b}")
            self.assertEqual(borrow, exp_borrow, f"Mismatch borrow for {a} - {b}")

    def test_subtraction_with_borrow(self) -> None:
        cases = [
            (3, 10, (3 - 10) & MASK32, 1),
            (0, 1, MASK32, 1),
            (0, MASK32, 1, 1),
            (0x11111111, 0x12345678, (0x11111111 - 0x12345678) & MASK32, 1),
            (1, MASK32, 2, 1),
        ]
        for a, b, exp_diff, exp_borrow in cases:
            diff, borrow = _run_sub_co_u32(a, b)
            self.assertEqual(diff, exp_diff, f"Borrow mismatch diff for {a} - {b}")
            self.assertEqual(borrow, exp_borrow, f"Borrow flag mismatch for {a} - {b}")

    def test_alternating_bit_patterns(self) -> None:
        a, b = 0xAAAAAAAA, 0x55555555
        diff, borrow = _run_sub_co_u32(a, b)
        self.assertEqual(diff, 0x55555555)
        self.assertEqual(borrow, 0)

        diff2, borrow2 = _run_sub_co_u32(b, a)
        self.assertEqual(diff2, (0x55555555 - 0xAAAAAAAA) & MASK32)
        self.assertEqual(borrow2, 1)


class TestValuSubCoU32OperandsAndRouting(unittest.TestCase):
    """Operand flexibility: SGPR destinations, null destination, SGPR and literal sources."""

    def test_borrow_into_sgpr_destination(self) -> None:
        # Borrow routed into s8
        diff, borrow = _run_sub_co_u32(10, 25, sdst="s8")
        self.assertEqual(diff, (10 - 25) & MASK32)
        self.assertEqual(borrow, 1)

        diff_noborrow, borrow_noborrow = _run_sub_co_u32(25, 10, sdst="s8")
        self.assertEqual(diff_noborrow, 15)
        self.assertEqual(borrow_noborrow, 0)

    def test_null_sdst_discards_borrow(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v1, 45",
            "v_mov_b32 v2, 15",
            "v_sub_co_u32 v0, null, v1, v2",
            "exp mrt0 v0, 0, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_null_sdst_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 30)

    def test_sgpr_src0_and_literal_src1(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "s_mov_b32 s2, 75",
            "v_sub_co_u32 v0, vcc_lo, s2, 25",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_sgpr_src_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 50)
        self.assertEqual(words[1] & 1, 0)

    def test_vgpr_src0_and_literal_src1(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v1, 100",
            "v_sub_co_u32 v0, vcc_lo, v1, 30",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_lit_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 70)
        self.assertEqual(words[1] & 1, 0)


class TestValuSubCoU32RegisterAliasing(unittest.TestCase):
    """In-place destination and source register aliasing scenarios."""

    def test_inplace_dst_equals_src0(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 40",
            "v_sub_co_u32 v0, vcc_lo, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_alias_s0_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 60)
        self.assertEqual(words[1] & 1, 0)

    def test_inplace_dst_equals_src1(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 80",
            "v_sub_co_u32 v1, vcc_lo, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v1, v3, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_alias_s1_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, (50 - 80) & MASK32)
        self.assertEqual(words[1] & 1, 1)

    def test_inplace_all_aliased(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v0, 0xdeadbeef",
            "v_sub_co_u32 v0, vcc_lo, v0, v0",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_alias_all_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0)
        self.assertEqual(words[1] & 1, 0)


class TestValuSubCoU32ClampAndMultiPrecision(unittest.TestCase):
    """Output clamp modifier and multi-precision 64-bit borrow chain execution."""

    def test_clamp_modifier_saturates_to_zero_on_borrow(self) -> None:
        # 10 - 20 with clamp => saturates to 0, vcc_lo borrow is 1
        diff, borrow = _run_sub_co_u32(10, 20, clamp=True)
        self.assertEqual(diff, 0, "Clamped subtraction with borrow must saturate to 0")
        self.assertEqual(borrow, 1, "Carry/borrow flag must still reflect true borrow under clamp")

    def test_clamp_modifier_without_borrow(self) -> None:
        # 30 - 12 with clamp => 18, vcc_lo borrow is 0
        diff, borrow = _run_sub_co_u32(30, 12, clamp=True)
        self.assertEqual(diff, 18)
        self.assertEqual(borrow, 0)

    def test_multi_precision_64bit_borrow_chain(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        # 64-bit subtraction chain:
        # A = 0x0000_0002_0000_0001, B = 0x0000_0001_0000_0005
        # Lower: 1 - 5 = 0xFFFF_FFFC with borrow = 1
        # Upper: 2 - 1 - borrow = 0 with borrow = 0
        a_lo, a_hi = 1, 2
        b_lo, b_hi = 5, 1
        lines = [
            f"v_mov_b32 v0, {a_lo}",
            f"v_mov_b32 v1, {a_hi}",
            f"v_mov_b32 v2, {b_lo}",
            f"v_mov_b32 v3, {b_hi}",
            "v_sub_co_u32 v4, vcc_lo, v0, v2",
            "v_sub_co_ci_u32 v5, vcc_lo, v1, v3, vcc_lo",
            "exp mrt0 v4, v5, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_chain_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0xFFFF_FFFC)
        self.assertEqual(words[1] & MASK32, 0)


class TestValuSubCoU32BoundaryExtremesAndBitpatterns(unittest.TestCase):
    """Integer limits, sign-bit transitions, and IEEE 754 bit-pattern invariance."""

    def test_integer_boundary_extremes(self) -> None:
        boundary_cases = [
            (MASK32, MASK32, 0, 0),
            (MASK32, 0, MASK32, 0),
            (0, MASK32, 1, 1),
            (0x8000_0000, 1, 0x7FFF_FFFF, 0),
            (0x7FFF_FFFF, 0x8000_0000, MASK32, 1),
        ]
        for src0, src1, exp_diff, exp_borrow in boundary_cases:
            diff, borrow = _run_sub_co_u32(src0, src1)
            self.assertEqual(diff, exp_diff, f"Diff mismatch for {src0:#x} - {src1:#x}")
            self.assertEqual(borrow, exp_borrow, f"Borrow mismatch for {src0:#x} - {src1:#x}")

    def test_float_bitpatterns_treated_as_integers(self) -> None:
        pos_inf = struct.unpack(">I", struct.pack(">f", float("inf")))[0]
        neg_inf = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]
        qnan = struct.unpack(">I", struct.pack(">f", float("nan")))[0]

        diff_inf, borrow_inf = _run_sub_co_u32(pos_inf, neg_inf)
        self.assertEqual(diff_inf, (pos_inf - neg_inf) & MASK32)
        self.assertEqual(borrow_inf, 1 if pos_inf < neg_inf else 0)

        diff_nan, borrow_nan = _run_sub_co_u32(qnan, 0)
        self.assertEqual(diff_nan, qnan)
        self.assertEqual(borrow_nan, 0)


class TestValuSubCoU32ConditionCodesAndPreservation(unittest.TestCase):
    """Condition code flags and scalar register preservation."""

    def test_vcc_predicate_use_and_preservation(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 50",
            "v_sub_co_u32 v0, vcc_lo, v1, v2",     # 100 - 50 = 50, vcc_lo = 0
            "v_cndmask_b32 v3, 0x1111, 0x2222, vcc_lo",  # Picks 0x1111
            "exp mrt0 v0, v3, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_vcc_pred_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 50)
        self.assertEqual(words[1] & MASK32, 0x1111)

    def test_scc_flag_preservation(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "s_cmp_eq_u32 5, 5",                          # SCC = 1
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 10",
            "v_sub_co_u32 v2, vcc_lo, v0, v1",           # VALU op executes
            "s_cselect_b32 s0, 0xAAAA, 0xBBBB",           # Picks 0xAAAA because SCC is 1
            "v_mov_b32 v3, s0",
            "exp mrt0 v2, v3, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_scc_pres_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 10)
        self.assertEqual(words[1] & MASK32, 0xAAAA)


class TestValuSubCoU32ReviewerMutationsAndRejection(unittest.TestCase):
    """Reviewer mutation pins and negative syntax/operand controls."""

    def test_translator_emits_op_isub_borrow(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 5",
            "v_sub_co_u32 v0, vcc_lo, v1, v2",
            "exp mrt0 v0, 0, 0, 0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_emit_check_{_TRANSLATE_ID}", lines)
        self.assertIn("OpISubBorrow", spv)
        self.assertNotIn("OpIAddCarry", spv)
        self.assertIn("OpCompositeExtract %uint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 15)

    def test_mutation_pin_unsigned_vs_signed_borrow(self) -> None:
        """Pin unsigned borrow vs signed borrow/overflow.

        For 0x8000_0000 - 1:
        - Unsigned subtraction: 0x8000_0000 >= 1, so borrow is 0.
        - Signed subtraction: -2147483648 - 1 causes signed underflow/overflow.
        The instruction must emit an unsigned borrow (0).
        """
        diff, borrow = _run_sub_co_u32(0x8000_0000, 1)
        self.assertEqual(diff, 0x7FFF_FFFF)
        self.assertEqual(borrow, 0)
        self.assertNotEqual(borrow, 1)

    def test_syntax_rejection_bare_opcode(self) -> None:
        lines = ["v_sub_co_u32", "s_endpgm"]
        with self.assertRaises((IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_sub_bad_bare", lines)

    def test_syntax_rejection_missing_operands(self) -> None:
        lines = ["v_sub_co_u32 v0", "s_endpgm"]
        with self.assertRaises((IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_sub_bad_missing", lines)

    def test_syntax_rejection_invalid_opcode(self) -> None:
        lines = ["v_sub_co_u32_invalid v0, vcc_lo, v1, v2", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_sub_bad_opcode", lines)

    def test_syntax_rejection_invalid_modifier(self) -> None:
        lines = ["v_sub_co_u32 v0, vcc_lo, v1, v2 invalid_mod", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_sub_bad_mod", lines)


if __name__ == "__main__":
    unittest.main()
