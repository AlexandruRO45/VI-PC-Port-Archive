from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines

MASK32 = 0xFFFFFFFF


def run_ps(lines: list[str], exp: tuple[str, str, str, str] = ("v0", "v1", "v2", "v3")) -> list[int]:
    full_lines = list(lines) + [
        f"exp mrt0 {exp[0]}, {exp[1]}, {exp[2]}, {exp[3]} done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_salu_bitscan", full_lines)
    return mrt0_words(spv)


class TestSFf0I32B32(unittest.TestCase):
    def test_scan_cases(self):
        lines = [
            "s_ff0_i32_b32 s0, 0",
            "s_ff0_i32_b32 s1, 1",
            "s_ff0_i32_b32 s2, 7",
            "s_ff0_i32_b32 s3, 0x7FFFFFFF",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0, 1, 3, 31])

    def test_all_ones_returns_mask32(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_ff0_i32_b32 s1, s0",
            "v_mov_b32 v0, s1",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [MASK32, MASK32, MASK32, MASK32])

    def test_isolated_zero_bits(self):
        lines = [
            f"s_mov_b32 s0, {(~1) & MASK32:#x}",
            "s_ff0_i32_b32 s4, s0",
            f"s_mov_b32 s1, {(~(1 << 8)) & MASK32:#x}",
            "s_ff0_i32_b32 s5, s1",
            f"s_mov_b32 s2, {(~(1 << 16)) & MASK32:#x}",
            "s_ff0_i32_b32 s6, s2",
            f"s_mov_b32 s3, {(~(1 << 24)) & MASK32:#x}",
            "s_ff0_i32_b32 s7, s3",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, s7",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [0, 8, 16, 24])

    def test_alternating_patterns(self):
        lines = [
            "s_mov_b32 s0, 0x55555555",
            "s_ff0_i32_b32 s2, s0",
            "s_mov_b32 s1, 0xAAAAAAAA",
            "s_ff0_i32_b32 s3, s1",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])

    def test_dest_aliasing(self):
        lines = [
            "s_mov_b32 s0, 1",
            "s_ff0_i32_b32 s0, s0",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [1, 1, 1, 1])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s0, 1",
            "s_ff0_i32_b32 s2, s0",
            "s_cselect_b32 s4, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_mov_b32 s1, 1",
            "s_ff0_i32_b32 s3, s1",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


class TestSFf0I32B64(unittest.TestCase):
    def test_scan_cases(self):
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 0",
            "s_ff0_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_mov_b32 s3, 0",
            "s_ff0_i32_b64 s5, s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [0, 32, 0, 32])

    def test_high_dword_zeros_and_all_ones(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            f"s_mov_b32 s1, {(~(1 << 8)) & MASK32:#x}",
            "s_ff0_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0xFFFFFFFF",
            f"s_mov_b32 s3, {(~(1 << 31)) & MASK32:#x}",
            "s_ff0_i32_b64 s5, s[2:3]",
            "s_mov_b32 s6, 0xFFFFFFFF",
            "s_mov_b32 s7, 0xFFFFFFFF",
            "s_ff0_i32_b64 s8, s[6:7]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s8",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [40, 63, MASK32, 40])

    def test_dword_boundary_transitions(self):
        lines = [
            f"s_mov_b32 s0, {(~(1 << 31)) & MASK32:#x}",
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_ff0_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0xFFFFFFFF",
            f"s_mov_b32 s3, {(~1) & MASK32:#x}",
            "s_ff0_i32_b64 s5, s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [31, 32, 31, 32])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 0",
            "s_ff0_i32_b64 s4, s[0:1]",
            "s_cselect_b32 s6, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_ff0_i32_b64 s5, s[0:1]",
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


class TestSFf1I32B32(unittest.TestCase):
    def test_scan_cases(self):
        lines = [
            "s_ff1_i32_b32 s0, 0",
            "s_ff1_i32_b32 s1, 1",
            "s_ff1_i32_b32 s2, 0x12",
            "s_ff1_i32_b32 s3, 0x80000000",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [MASK32, 0, 1, 31])

    def test_powers_of_two(self):
        lines = [
            "s_mov_b32 s0, 0x00000010",
            "s_ff1_i32_b32 s4, s0",
            "s_mov_b32 s1, 0x00001000",
            "s_ff1_i32_b32 s5, s1",
            "s_mov_b32 s2, 0x00100000",
            "s_ff1_i32_b32 s6, s2",
            "s_mov_b32 s3, 0x10000000",
            "s_ff1_i32_b32 s7, s3",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, s7",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [4, 12, 20, 28])

    def test_dest_aliasing(self):
        lines = [
            "s_mov_b32 s0, 0x80",
            "s_ff1_i32_b32 s0, s0",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [7, 7, 7, 7])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s0, 1",
            "s_ff1_i32_b32 s2, s0",
            "s_cselect_b32 s4, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_mov_b32 s1, 1",
            "s_ff1_i32_b32 s3, s1",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


class TestSFf1I32B64(unittest.TestCase):
    def test_scan_cases(self):
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 0",
            "s_ff1_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 1",
            "s_mov_b32 s3, 0",
            "s_ff1_i32_b64 s5, s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [MASK32, 0, MASK32, 0])

    def test_high_dword_bits(self):
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 1",
            "s_ff1_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0",
            f"s_mov_b32 s3, {1 << 23:#x}",
            "s_ff1_i32_b64 s5, s[2:3]",
            "s_mov_b32 s6, 0",
            "s_mov_b32 s7, 0x80000000",
            "s_ff1_i32_b64 s8, s[6:7]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s8",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [32, 55, 63, 32])

    def test_dword_boundary_transitions(self):
        lines = [
            "s_mov_b32 s0, 0x80000000",
            "s_mov_b32 s1, 0",
            "s_ff1_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 1",
            "s_ff1_i32_b64 s5, s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [31, 32, 31, 32])

    def test_mixed_low_and_high(self):
        lines = [
            f"s_mov_b32 s0, {1 << 4:#x}",
            f"s_mov_b32 s1, {1 << 16:#x}",
            "s_ff1_i32_b64 s2, s[0:1]",
            "v_mov_b32 v0, s2",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [4, 4, 4, 4])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s0, 1",
            "s_mov_b32 s1, 0",
            "s_ff1_i32_b64 s4, s[0:1]",
            "s_cselect_b32 s6, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_ff1_i32_b64 s5, s[0:1]",
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


class TestSFlbitI32B32(unittest.TestCase):
    def test_scan_cases(self):
        lines = [
            "s_flbit_i32_b32 s0, 0",
            "s_mov_b32 s4, 0x80000000",
            "s_flbit_i32_b32 s1, s4",
            "s_mov_b32 s5, 0x40000000",
            "s_flbit_i32_b32 s2, s5",
            "s_flbit_i32_b32 s3, 1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [MASK32, 0, 1, 31])

    def test_powers_of_two(self):
        lines = [
            "s_mov_b32 s0, 0x00010000",
            "s_flbit_i32_b32 s4, s0",
            "s_mov_b32 s1, 0x01000000",
            "s_flbit_i32_b32 s5, s1",
            "s_mov_b32 s2, 0x00000100",
            "s_flbit_i32_b32 s6, s2",
            "s_mov_b32 s3, 0x00FF0000",
            "s_flbit_i32_b32 s7, s3",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s6",
            "v_mov_b32 v3, s7",
        ]
        words = run_ps(lines)
        self.assertEqual(words, [15, 7, 23, 8])

    def test_dest_aliasing(self):
        lines = [
            "s_mov_b32 s0, 0x00010000",
            "s_flbit_i32_b32 s0, s0",
            "v_mov_b32 v0, s0",
        ]
        words = run_ps(lines, ("v0", "v0", "v0", "v0"))
        self.assertEqual(words, [15, 15, 15, 15])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_flbit_i32_b32 s0, 1",
            "s_cselect_b32 s2, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_flbit_i32_b32 s1, 1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


class TestSFlbitI32B64(unittest.TestCase):
    def test_scan_cases(self):
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 0",
            "s_flbit_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0x80000000",
            "s_flbit_i32_b64 s5, s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [MASK32, 0, MASK32, 0])

    def test_high_and_low_dword_leading_zeros(self):
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 0x40000000",
            "s_flbit_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 1",
            "s_flbit_i32_b64 s5, s[2:3]",
            "s_mov_b32 s6, 1",
            "s_mov_b32 s7, 0",
            "s_flbit_i32_b64 s8, s[6:7]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "v_mov_b32 v2, s8",
        ]
        words = run_ps(lines, ("v0", "v1", "v2", "v0"))
        self.assertEqual(words, [1, 31, 63, 1])

    def test_dword_boundary_transitions(self):
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 1",
            "s_flbit_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0x80000000",
            "s_mov_b32 s3, 0",
            "s_flbit_i32_b64 s5, s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [31, 32, 31, 32])

    def test_mixed_low_and_high(self):
        lines = [
            f"s_mov_b32 s0, {1 << 10:#x}",
            f"s_mov_b32 s1, {1 << (60 - 32):#x}",
            "s_flbit_i32_b64 s4, s[0:1]",
            "s_mov_b32 s2, 0xFFFFFFFF",
            "s_mov_b32 s3, 0",
            "s_flbit_i32_b64 s5, s[2:3]",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [3, 32, 3, 32])

    def test_scc_preserved(self):
        lines = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s0, 1",
            "s_mov_b32 s1, 0",
            "s_flbit_i32_b64 s4, s[0:1]",
            "s_cselect_b32 s6, 1, 0",
            "s_cmp_lg_u32 1, 1",
            "s_flbit_i32_b64 s5, s[0:1]",
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
        ]
        words = run_ps(lines, ("v0", "v1", "v0", "v1"))
        self.assertEqual(words, [1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
