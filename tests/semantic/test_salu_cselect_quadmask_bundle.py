from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines


def run_ps(lines: list[str], exp: tuple[str, str, str, str] = ("v0", "v1", "v2", "v3")) -> list[int]:
    full_lines = list(lines) + [
        f"exp mrt0 {exp[0]}, {exp[1]}, {exp[2]}, {exp[3]} done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_salu_bundle", full_lines)
    return mrt0_words(spv)


class TestSCSelectB32(unittest.TestCase):
    def test_scc_set_selects_first(self):
        lines = [
            "s_cmp_eq_u32 5, 5",
            "s_cselect_b32 s0, 0x12345678, 0x87654321",
            "s_cselect_b32 s1, 0xFFFFFFFF, 0x00000000",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x12345678, 0xFFFFFFFF, 0x12345678, 0xFFFFFFFF])

    def test_scc_cleared_selects_second(self):
        lines = [
            "s_cmp_lg_u32 5, 5",
            "s_cselect_b32 s0, 0x12345678, 0x87654321",
            "s_cselect_b32 s1, 0xFFFFFFFF, 0x00000000",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x87654321, 0x00000000, 0x87654321, 0x00000000])

    def test_extremum_constants(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b32 s0, 0x80000000, 0x7FFFFFFF",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b32 s1, 0x80000000, 0x7FFFFFFF",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x80000000, 0x7FFFFFFF, 0x80000000, 0x7FFFFFFF])

    def test_identical_operands(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b32 s0, 0xAAAA5555, 0xAAAA5555",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b32 s1, 0xAAAA5555, 0xAAAA5555",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0xAAAA5555, 0xAAAA5555, 0xAAAA5555, 0xAAAA5555])

    def test_dest_aliasing(self):
        lines = [
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b32 s0, s0, s1",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b32 s1, s0, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x11111111, 0x22222222, 0x11111111, 0x22222222])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b32 s0, 0x10, 0x20",
            "s_cselect_b32 s1, 0x30, 0x40",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b32 s2, 0x10, 0x20",
            "s_cselect_b32 s3, 0x30, 0x40",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x10, 0x30, 0x20, 0x40])


class TestSCSelectB64(unittest.TestCase):
    def test_scc_set_selects_first_pair(self):
        lines = [
            "s_mov_b32 s2, 0x11112222",
            "s_mov_b32 s3, 0x33334444",
            "s_mov_b32 s4, 0x55556666",
            "s_mov_b32 s5, 0x77778888",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 s[0:1], s[2:3], s[4:5]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x11112222, 0x33334444, 0x11112222, 0x33334444])

    def test_scc_cleared_selects_second_pair(self):
        lines = [
            "s_mov_b32 s2, 0x11112222",
            "s_mov_b32 s3, 0x33334444",
            "s_mov_b32 s4, 0x55556666",
            "s_mov_b32 s5, 0x77778888",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b64 s[0:1], s[2:3], s[4:5]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x55556666, 0x77778888, 0x55556666, 0x77778888])

    def test_64bit_boundary_extremes(self):
        lines = [
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_mov_b32 s3, 0xFFFFFFFF",
            "s_mov_b32 s4, 0x00000000",
            "s_mov_b32 s5, 0x00000000",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 s[0:1], s[2:3], s[4:5]",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b64 s[6:7], s[2:3], s[4:5]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, s7",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000])

    def test_dest_aliasing_inplace(self):
        lines = [
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_mov_b32 s2, 0x33333333",
            "s_mov_b32 s3, 0x44444444",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 s[0:1], s[0:1], s[2:3]",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b64 s[2:3], s[0:1], s[2:3]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x11111111, 0x22222222, 0x33333333, 0x44444444])

    def test_scc_preserved(self):
        lines = [
            "s_mov_b32 s2, 0xAAAA0000",
            "s_mov_b32 s3, 0x0000BBBB",
            "s_mov_b32 s4, 0xCCCC0000",
            "s_mov_b32 s5, 0x0000DDDD",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 s[0:1], s[2:3], s[4:5]",
            "s_cselect_b32 s6, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_cselect_b64 s[0:1], s[2:3], s[4:5]",
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


class TestSQuadmaskB32(unittest.TestCase):
    def test_alternating_nibbles(self):
        lines = [
            "s_mov_b32 s2, 0x01020408",
            "s_quadmask_b32 s0, s2",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x55, 0x1, 0x55, 0x1])

    def test_zero_input(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s2, 0",
            "s_quadmask_b32 s0, s2",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x0, 0x0, 0x0, 0x0])

    def test_all_ones(self):
        lines = [
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_quadmask_b32 s0, s2",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0xFF, 0x1, 0xFF, 0x1])

    def test_single_bit_in_each_nibble(self):
        lines = [
            "s_mov_b32 s2, 0x00000001",
            "s_quadmask_b32 s0, s2",
            "s_mov_b32 s3, 0x00000020",
            "s_quadmask_b32 s1, s3",
            "s_mov_b32 s4, 0x00000400",
            "s_quadmask_b32 s5, s4",
            "s_mov_b32 s6, 0x00008000",
            "s_quadmask_b32 s7, s6",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s5",
            "v_mov_b32 v3, s7",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0x1, 0x2, 0x4, 0x8])

    def test_upper_bits_strictly_zero(self):
        lines = [
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_quadmask_b32 s0, s2",
            "s_and_b32 s1, s0, 0xFFFFFF00",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0xFF, 0x0, 0xFF, 0x0])


class TestSQuadmaskB64(unittest.TestCase):
    def test_nonzero_64bit(self):
        lines = [
            "s_mov_b32 s2, 0x01020408",
            "s_mov_b32 s3, 0x01020408",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "s_cselect_b32 s2, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [0x5555, 0, 1, 0x5555])

    def test_zero_64bit(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "s_cselect_b32 s2, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [0, 0, 0, 0])

    def test_high_word_only(self):
        lines = [
            "s_mov_b32 s2, 0x00000000",
            "s_mov_b32 s3, 0xF0000000",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "s_cselect_b32 s2, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [0x8000, 0, 1, 0x8000])

    def test_low_word_only(self):
        lines = [
            "s_mov_b32 s2, 0x0000000F",
            "s_mov_b32 s3, 0x00000000",
            "s_quadmask_b64 s[0:1], s[2:3]",
            "s_cselect_b32 s2, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [0x0001, 0, 1, 0x0001])

    def test_dest_aliasing(self):
        lines = [
            "s_mov_b32 s0, 0x01020408",
            "s_mov_b32 s1, 0x01020408",
            "s_quadmask_b64 s[0:1], s[0:1]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x5555, 0, 0x5555, 0])


class TestSBitset0B32(unittest.TestCase):
    def test_clear_specific_bits(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_bitset0_b32 s0, 4",
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_bitset0_b32 s1, 0",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_bitset0_b32 s2, 31",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [0xFFFFFFEF, 0xFFFFFFFE, 0x7FFFFFFF, 0xFFFFFFEF])

    def test_clear_modulo_32(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s4, 36",
            "s_bitset0_b32 s0, s4",
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s5, 64",
            "s_bitset0_b32 s1, s5",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0xFFFFFFEF, 0xFFFFFFFE, 0xFFFFFFEF, 0xFFFFFFFE])

    def test_already_cleared_bit(self):
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_bitset0_b32 s0, 10",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [0, 0, 0, 0])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_bitset0_b32 s0, 4",
            "s_cselect_b32 s1, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_bitset0_b32 s2, 4",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s1",
            "v_mov_b32 v1, s3",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


class TestSBitset1B32(unittest.TestCase):
    def test_set_specific_bits(self):
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_bitset1_b32 s0, 4",
            "s_mov_b32 s1, 0x00000000",
            "s_bitset1_b32 s1, 0",
            "s_mov_b32 s2, 0x00000000",
            "s_bitset1_b32 s2, 31",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [0x00000010, 0x00000001, 0x80000000, 0x00000010])

    def test_set_modulo_32(self):
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_mov_b32 s4, 36",
            "s_bitset1_b32 s0, s4",
            "s_mov_b32 s1, 0x00000000",
            "s_mov_b32 s5, 64",
            "s_bitset1_b32 s1, s5",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0x00000010, 0x00000001, 0x00000010, 0x00000001])

    def test_already_set_bit(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_bitset1_b32 s0, 15",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF])

    def test_accumulate_bits(self):
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_bitset1_b32 s0, 1",
            "s_bitset1_b32 s0, 3",
            "s_bitset1_b32 s0, 7",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [0x8A, 0x8A, 0x8A, 0x8A])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s0, 0x00000000",
            "s_bitset1_b32 s0, 4",
            "s_cselect_b32 s1, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_mov_b32 s2, 0x00000000",
            "s_bitset1_b32 s2, 4",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s1",
            "v_mov_b32 v1, s3",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
