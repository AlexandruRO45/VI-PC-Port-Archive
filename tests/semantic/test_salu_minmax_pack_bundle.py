"""Semantic reference tests and translator-backed validation for AMD RDNA2 SALU Min/Max & Pack Suite.

Subsystem Feature Bundle 32 covers 6 SALU min/max selection and half-word packing instructions:
  1. s_min_i32         - Signed 32-bit minimum: D = min(S0, S1), SCC = (S0 < S1, signed)
  2. s_min_u32         - Unsigned 32-bit minimum: D = min(S0, S1), SCC = (S0 < S1, unsigned)
  3. s_max_i32         - Signed 32-bit maximum: D = max(S0, S1), SCC = (S0 > S1, signed)
  4. s_max_u32         - Unsigned 32-bit maximum: D = max(S0, S1), SCC = (S0 > S1, unsigned)
  5. s_pack_ll_b32_b16 - Pack low halves: D = {S1[15:0], S0[15:0]}, SCC untouched
  6. s_pack_hh_b32_b16 - Pack high halves: D = {S1[31:16], S0[31:16]}, SCC untouched

RDNA2 SALU Architecture:
  - Operates on 32-bit scalar registers (SGPRs), inline/literal constants.
  - Condition code (SCC):
      s_min_i32: SCC = 1 if S0 < S1 (signed), 0 on >= (ties set SCC to 0)
      s_min_u32: SCC = 1 if S0 < S1 (unsigned), 0 on >= (ties set SCC to 0)
      s_max_i32: SCC = 1 if S0 > S1 (signed), 0 on <= (ties set SCC to 0)
      s_max_u32: SCC = 1 if S0 > S1 (unsigned), 0 on <= (ties set SCC to 0)
      s_pack_*: SCC is preserved (untouched)
  - 100% of tests are wired to end-to-end SPIR-V assembly translation through tools/rdna2spirv.py.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.path.join(ROOT, "tools")
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import mrt0_words, translate_lines  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF


def run_salu(
    op: str,
    s0: int,
    s1: int,
    *,
    has_scc: bool = True,
    name: str = "",
    dst: str = "s2",
    src0: str = "s0",
    src1: str = "s1",
) -> tuple[int, int]:
    """Execute SALU instruction `op <dst>, <src0>, <src1>` via translate_lines and mrt0_words.

    Returns (dst_word, scc_word).
    """
    if not name:
        name = f"ps_{op}_{s0 & MASK32:#x}_{s1 & MASK32:#x}"
    lines = [
        f"s_mov_b32 s0, {s0 & MASK32:#010x}",
        f"s_mov_b32 s1, {s1 & MASK32:#010x}",
        f"{op} {dst}, {src0}, {src1}",
    ]
    if has_scc:
        lines += [
            "s_cselect_b32 s3, 1, 0",
            f"v_mov_b32 v0, {dst}",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
        ]
    else:
        lines += [
            f"v_mov_b32 v0, {dst}",
            "exp mrt0 v0, v0, v0, v0 done vm",
        ]
    lines.append("s_endpgm")
    spv = translate_lines(name, lines)
    words = mrt0_words(spv)
    return words[0], (words[1] if has_scc else 0)


def run_pack_with_prior_scc(op: str, s0: int, s1: int, prior_scc: int, name: str = "") -> tuple[int, int]:
    """Execute s_pack_* instruction while verifying preservation of prior SCC state."""
    if not name:
        name = f"ps_{op}_scc_{prior_scc}_{s0 & MASK32:#x}_{s1 & MASK32:#x}"
    # Set prior SCC to 1 via s_cmp_eq_u32 1, 1 or 0 via s_cmp_eq_u32 1, 0
    lines = [
        f"s_cmp_eq_u32 1, {1 if prior_scc else 0}",
        f"s_mov_b32 s0, {s0 & MASK32:#010x}",
        f"s_mov_b32 s1, {s1 & MASK32:#010x}",
        f"{op} s2, s0, s1",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32 v0, s2",
        "v_mov_b32 v1, s3",
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(name, lines)
    words = mrt0_words(spv)
    return words[0], words[1]


# =============================================================================
# Opcode 1: s_min_i32
# =============================================================================
class TestSMinI32(unittest.TestCase):
    """Translator-backed semantic tests for s_min_i32 (Signed 32-bit Minimum)."""

    def test_min_i32_basic_positive(self):
        val, scc = run_salu("s_min_i32", 10, 20, name="ps_min_i32_pos")
        self.assertEqual(val, 10)
        self.assertEqual(scc, 1)

    def test_min_i32_negative_numbers(self):
        # min(-20, 10) = -20, SCC = 1 (-20 < 10)
        val, scc = run_salu("s_min_i32", -20 & MASK32, 10, name="ps_min_i32_neg1")
        self.assertEqual(val, -20 & MASK32)
        self.assertEqual(scc, 1)

        # min(-5, -15) = -15, SCC = 0 (-5 < -15 is False)
        val2, scc2 = run_salu("s_min_i32", -5 & MASK32, -15 & MASK32, name="ps_min_i32_neg2")
        self.assertEqual(val2, -15 & MASK32)
        self.assertEqual(scc2, 0)

    def test_min_i32_ties(self):
        # On tie: SCC must be 0
        val, scc = run_salu("s_min_i32", 100, 100, name="ps_min_i32_tie")
        self.assertEqual(val, 100)
        self.assertEqual(scc, 0)

    def test_min_i32_scc_extrema(self):
        # 0x80000000 (-2147483648) vs 0x7FFFFFFF (2147483647)
        val, scc = run_salu("s_min_i32", 0x80000000, 0x7FFFFFFF, name="ps_min_i32_extrema")
        self.assertEqual(val, 0x80000000)
        self.assertEqual(scc, 1)

    def test_min_i32_in_place_aliasing(self):
        # s_min_i32 s0, s0, s1 (destination aliases source 0)
        val, scc = run_salu("s_min_i32", 50, 25, dst="s0", src0="s0", src1="s1", name="ps_min_i32_alias")
        self.assertEqual(val, 25)
        self.assertEqual(scc, 0)

    def test_min_i32_spirv_lowering_and_execution(self):
        lines = [
            "s_mov_b32 s0, -5",
            "s_mov_b32 s1, 10",
            "s_min_i32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min_i32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFFFFFB)
        self.assertEqual(words[1], 1)

    def test_min_i32_syntax_rejection(self):
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_min_i32", ["s_min_i32 s0, s1", "s_endpgm"])


# =============================================================================
# Opcode 2: s_min_u32
# =============================================================================
class TestSMinU32(unittest.TestCase):
    """Translator-backed semantic tests for s_min_u32 (Unsigned 32-bit Minimum)."""

    def test_min_u32_basic(self):
        val, scc = run_salu("s_min_u32", 10, 20, name="ps_min_u32_basic")
        self.assertEqual(val, 10)
        self.assertEqual(scc, 1)

    def test_min_u32_unsigned_comparison(self):
        # 0xFFFFFFFF (4294967295) vs 10: unsigned min is 10, SCC = 0
        val, scc = run_salu("s_min_u32", 0xFFFFFFFF, 10, name="ps_min_u32_ucomp1")
        self.assertEqual(val, 10)
        self.assertEqual(scc, 0)

        # 10 vs 0xFFFFFFFF: unsigned min is 10, SCC = 1
        val2, scc2 = run_salu("s_min_u32", 10, 0xFFFFFFFF, name="ps_min_u32_ucomp2")
        self.assertEqual(val2, 10)
        self.assertEqual(scc2, 1)

    def test_min_u32_ties(self):
        val, scc = run_salu("s_min_u32", 100, 100, name="ps_min_u32_ties")
        self.assertEqual(val, 100)
        self.assertEqual(scc, 0)

    def test_min_u32_extrema(self):
        val, scc = run_salu("s_min_u32", 0, 0xFFFFFFFF, name="ps_min_u32_extrema")
        self.assertEqual(val, 0)
        self.assertEqual(scc, 1)

    def test_min_u32_in_place_aliasing(self):
        val, scc = run_salu("s_min_u32", 0x80000000, 0x40000000, dst="s1", src0="s0", src1="s1", name="ps_min_u32_alias")
        self.assertEqual(val, 0x40000000)
        self.assertEqual(scc, 0)

    def test_min_u32_spirv_lowering_and_execution(self):
        lines = [
            "s_mov_b32 s0, 10",
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_min_u32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min_u32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 10)
        self.assertEqual(words[1], 1)

    def test_min_u32_syntax_rejection(self):
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_min_u32", ["s_min_u32 s0, s1", "s_endpgm"])


# =============================================================================
# Opcode 3: s_max_i32
# =============================================================================
class TestSMaxI32(unittest.TestCase):
    """Translator-backed semantic tests for s_max_i32 (Signed 32-bit Maximum)."""

    def test_max_i32_basic(self):
        val, scc = run_salu("s_max_i32", 10, 20, name="ps_max_i32_basic")
        self.assertEqual(val, 20)
        self.assertEqual(scc, 0)  # 10 > 20 is False

    def test_max_i32_s0_greater(self):
        val, scc = run_salu("s_max_i32", 30, 20, name="ps_max_i32_s0_gt")
        self.assertEqual(val, 30)
        self.assertEqual(scc, 1)  # 30 > 20 is True

    def test_max_i32_negatives(self):
        # max(-5, -15) = -5, SCC = 1 (-5 > -15)
        val, scc = run_salu("s_max_i32", -5 & MASK32, -15 & MASK32, name="ps_max_i32_neg")
        self.assertEqual(val, -5 & MASK32)
        self.assertEqual(scc, 1)

    def test_max_i32_ties(self):
        val, scc = run_salu("s_max_i32", 100, 100, name="ps_max_i32_ties")
        self.assertEqual(val, 100)
        self.assertEqual(scc, 0)

    def test_max_i32_extrema(self):
        val, scc = run_salu("s_max_i32", 0x7FFFFFFF, 0x80000000, name="ps_max_i32_extrema")
        self.assertEqual(val, 0x7FFFFFFF)
        self.assertEqual(scc, 1)

    def test_max_i32_in_place_aliasing(self):
        val, scc = run_salu("s_max_i32", 10, 40, dst="s0", src0="s0", src1="s1", name="ps_max_i32_alias")
        self.assertEqual(val, 40)
        self.assertEqual(scc, 0)

    def test_max_i32_spirv_lowering_and_execution(self):
        lines = [
            "s_mov_b32 s0, -5",
            "s_mov_b32 s1, 10",
            "s_max_i32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max_i32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 10)
        self.assertEqual(words[1], 0)

    def test_max_i32_syntax_rejection(self):
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_max_i32", ["s_max_i32 s0", "s_endpgm"])


# =============================================================================
# Opcode 4: s_max_u32
# =============================================================================
class TestSMaxU32(unittest.TestCase):
    """Translator-backed semantic tests for s_max_u32 (Unsigned 32-bit Maximum)."""

    def test_max_u32_basic(self):
        val, scc = run_salu("s_max_u32", 10, 20, name="ps_max_u32_basic")
        self.assertEqual(val, 20)
        self.assertEqual(scc, 0)

    def test_max_u32_s0_greater(self):
        val, scc = run_salu("s_max_u32", 0xFFFFFFFF, 10, name="ps_max_u32_s0_gt")
        self.assertEqual(val, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_max_u32_ties(self):
        val, scc = run_salu("s_max_u32", 100, 100, name="ps_max_u32_ties")
        self.assertEqual(val, 100)
        self.assertEqual(scc, 0)

    def test_max_u32_extrema(self):
        val, scc = run_salu("s_max_u32", 0, 0xFFFFFFFF, name="ps_max_u32_extrema")
        self.assertEqual(val, 0xFFFFFFFF)
        self.assertEqual(scc, 0)

    def test_max_u32_in_place_aliasing(self):
        val, scc = run_salu("s_max_u32", 0xFFFFFFFF, 0x1000, dst="s1", src0="s0", src1="s1", name="ps_max_u32_alias")
        self.assertEqual(val, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_max_u32_spirv_lowering_and_execution(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s1, 10",
            "s_max_u32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max_u32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)

    def test_max_u32_syntax_rejection(self):
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_max_u32", ["s_max_u32 s0, s1", "s_endpgm"])


# =============================================================================
# Opcode 5: s_pack_ll_b32_b16
# =============================================================================
class TestSPackLlB32B16(unittest.TestCase):
    """Translator-backed semantic tests for s_pack_ll_b32_b16 (Pack Low Halves)."""

    def test_pack_ll_basic(self):
        # s0 = 0x12345678, s1 = 0xABCDEF01 -> {0xef01, 0x5678} = 0xEF015678
        val, _ = run_salu("s_pack_ll_b32_b16", 0x12345678, 0xABCDEF01, has_scc=False, name="ps_pack_ll_basic")
        self.assertEqual(val, 0xEF015678)

    def test_pack_ll_zeros_and_ones(self):
        val0, _ = run_salu("s_pack_ll_b32_b16", 0, 0, has_scc=False, name="ps_pack_ll_zeros")
        self.assertEqual(val0, 0)
        val1, _ = run_salu("s_pack_ll_b32_b16", 0xFFFFFFFF, 0xFFFFFFFF, has_scc=False, name="ps_pack_ll_ones")
        self.assertEqual(val1, 0xFFFFFFFF)

    def test_pack_ll_alternating_halves(self):
        val, _ = run_salu("s_pack_ll_b32_b16", 0xAAAA1111, 0xBBBB2222, has_scc=False, name="ps_pack_ll_alt")
        self.assertEqual(val, 0x22221111)

    def test_pack_ll_scc_preserved_when_set(self):
        val, scc = run_pack_with_prior_scc("s_pack_ll_b32_b16", 0x12345678, 0xABCDEF01, prior_scc=1, name="ps_pack_ll_scc1")
        self.assertEqual(val, 0xEF015678)
        self.assertEqual(scc, 1)

    def test_pack_ll_scc_preserved_when_clear(self):
        val, scc = run_pack_with_prior_scc("s_pack_ll_b32_b16", 0x12345678, 0xABCDEF01, prior_scc=0, name="ps_pack_ll_scc0")
        self.assertEqual(val, 0xEF015678)
        self.assertEqual(scc, 0)

    def test_pack_ll_in_place_aliasing(self):
        val, _ = run_salu("s_pack_ll_b32_b16", 0x12345678, 0xABCDEF01, has_scc=False, dst="s0", src0="s0", src1="s1", name="ps_pack_ll_alias")
        self.assertEqual(val, 0xEF015678)

    def test_pack_ll_spirv_lowering_and_execution(self):
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0xABCDEF01",
            "s_pack_ll_b32_b16 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_pack_ll_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 0xEF015678)

    def test_pack_ll_syntax_rejection(self):
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_pack_ll", ["s_pack_ll_b32_b16 s0, s1", "s_endpgm"])


# =============================================================================
# Opcode 6: s_pack_hh_b32_b16
# =============================================================================
class TestSPackHhB32B16(unittest.TestCase):
    """Translator-backed semantic tests for s_pack_hh_b32_b16 (Pack High Halves)."""

    def test_pack_hh_basic(self):
        # s0 = 0x12345678, s1 = 0xABCDEF01 -> {0xabcd, 0x1234} = 0xABCD1234
        val, _ = run_salu("s_pack_hh_b32_b16", 0x12345678, 0xABCDEF01, has_scc=False, name="ps_pack_hh_basic")
        self.assertEqual(val, 0xABCD1234)

    def test_pack_hh_zero_high_halves(self):
        val, _ = run_salu("s_pack_hh_b32_b16", 0x0000FFFF, 0x0000FFFF, has_scc=False, name="ps_pack_hh_zero_hi")
        self.assertEqual(val, 0)

    def test_pack_hh_alternating_halves(self):
        val, _ = run_salu("s_pack_hh_b32_b16", 0x43210000, 0x87650000, has_scc=False, name="ps_pack_hh_alt")
        self.assertEqual(val, 0x87654321)

    def test_pack_hh_scc_preserved_when_set(self):
        val, scc = run_pack_with_prior_scc("s_pack_hh_b32_b16", 0x12345678, 0xABCDEF01, prior_scc=1, name="ps_pack_hh_scc1")
        self.assertEqual(val, 0xABCD1234)
        self.assertEqual(scc, 1)

    def test_pack_hh_scc_preserved_when_clear(self):
        val, scc = run_pack_with_prior_scc("s_pack_hh_b32_b16", 0x12345678, 0xABCDEF01, prior_scc=0, name="ps_pack_hh_scc0")
        self.assertEqual(val, 0xABCD1234)
        self.assertEqual(scc, 0)

    def test_pack_hh_in_place_aliasing(self):
        val, _ = run_salu("s_pack_hh_b32_b16", 0x12345678, 0xABCDEF01, has_scc=False, dst="s0", src0="s0", src1="s1", name="ps_pack_hh_alias")
        self.assertEqual(val, 0xABCD1234)

    def test_pack_hh_spirv_lowering_and_execution(self):
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0xABCDEF01",
            "s_pack_hh_b32_b16 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_pack_hh_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 0xABCD1234)

    def test_pack_hh_syntax_rejection(self):
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_pack_hh", ["s_pack_hh_b32_b16 s0", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
