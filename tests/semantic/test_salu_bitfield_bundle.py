from __future__ import annotations

"""Semantic translator verification suite for RDNA2 SALU Bitfield Extract and Mask bundle.

Covers 6 opcodes with 100% translator execution via support.translate_lines and support.mrt0_words:
- s_bfe_u32: 32-bit unsigned bitfield extract (offset = src1[4:0], width = src1[22:16]), SCC = (dst != 0).
- s_bfe_i32: 32-bit signed bitfield extract, arithmetic sign extension from extracted MSB, SCC = (dst != 0).
- s_bfe_u64: 64-bit unsigned bitfield extract across SGPR pair, offset in [5:0], width in [22:16], SCC = (dst != 0).
- s_bfe_i64: 64-bit signed bitfield extract across SGPR pair, 64-bit arithmetic sign extension, SCC = (dst != 0).
- s_bfm_b32: 32-bit bitfield mask generator: D = ((1 << (src0 & 31)) - 1) << (src1 & 31).
- s_bfm_b64: 64-bit bitfield mask generator: D = ((1 << (src0 & 63)) - 1) << (src1 & 63) as lane mask.

Every test method compiles an RDNA2 assembly program, executes the translated SPIR-V through
spirv_mini_interp, and asserts the concrete register values written to mrt0.
Zero mock machines, zero game data, standard library Python only.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def run_ps_words(lines: list[str], name: str = "ps_salu_bitfield") -> list[int]:
    """Helper to assemble a pixel shader ending with mrt0 export and return the four 32-bit words."""
    full = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, full)
    return support.mrt0_words(spv)


# ==============================================================================
# 1. s_bfe_u32 Translator Tests
# ==============================================================================
class TestSBfeU32Translator(unittest.TestCase):
    def test_bfe_u32_canonical(self):
        # src0 = 0x12345678, width = 8, offset = 4 -> bits [11:4] = 0x67
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x00080004",  # width 8, offset 4
            "s_bfe_u32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",    # capture SCC
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u32_canon")
        self.assertEqual(words[0], 0x67)
        self.assertEqual(words[1], 1)

    def test_bfe_u32_zero_result_and_scc(self):
        # src0 = 0x12345608, width = 4, offset = 4 -> bits [7:4] = 0x0
        lines = [
            "s_mov_b32 s0, 0x12345608",
            "s_mov_b32 s1, 0x00040004",  # width 4, offset 4
            "s_bfe_u32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",    # capture SCC (should be 0)
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u32_zero")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)

    def test_bfe_u32_zero_width(self):
        # width = 0 -> output must be 0, SCC = 0
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s1, 0x00000004",  # width 0, offset 4
            "s_bfe_u32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u32_zwid")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)

    def test_bfe_u32_clamped_width(self):
        # offset 28, width 16 -> clamped to 32 - 28 = 4 bits
        # 0xF0000000 >> 28 = 0xF
        lines = [
            "s_mov_b32 s0, 0xF0000000",
            "s_mov_b32 s1, 0x0010001C",  # width 16, offset 28
            "s_bfe_u32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u32_clamp")
        self.assertEqual(words[0], 0xF)

    def test_bfe_u32_register_aliasing(self):
        # dst == src0
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x00080004",
            "s_bfe_u32 s0, s0, s1",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u32_alias")
        self.assertEqual(words[0], 0x67)

    def test_bfe_u32_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_bfe_u32_err", ["s_bfe_u32 s0, s1", "s_endpgm"])

    def test_bfe_u32_mutation_detection(self):
        lines = [
            "s_mov_b32 s0, 0x80",
            "s_mov_b32 s1, 0x00080000",  # width 8, offset 0 -> top bit is 1
            "s_bfe_u32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u32_mut")
        # u32 MUST NOT sign extend (should be 0x80, not 0xFFFFFF80)
        self.assertEqual(words[0], 0x80)
        self.assertNotEqual(words[0], 0xFFFFFF80)


# ==============================================================================
# 2. s_bfe_i32 Translator Tests
# ==============================================================================
class TestSBfeI32Translator(unittest.TestCase):
    def test_bfe_i32_negative_sign_extension(self):
        # src0 = 0x8, width = 4, offset = 0 -> bits [3:0] = 0x8 (MSB=1 -> negative -8)
        # Sign-extended to 32 bits -> 0xFFFFFFF8, SCC = 1
        lines = [
            "s_mov_b32 s0, 0x00000008",
            "s_mov_b32 s1, 0x00040000",
            "s_bfe_i32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i32_neg")
        self.assertEqual(words[0], 0xFFFFFFF8)
        self.assertEqual(words[1], 1)

    def test_bfe_i32_positive_sign_extension(self):
        # src0 = 0x7, width = 4, offset = 0 -> bits [3:0] = 0x7 (MSB=0 -> positive 7)
        # Sign-extended -> 0x00000007, SCC = 1
        lines = [
            "s_mov_b32 s0, 0x00000007",
            "s_mov_b32 s1, 0x00040000",
            "s_bfe_i32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i32_pos")
        self.assertEqual(words[0], 7)
        self.assertEqual(words[1], 1)

    def test_bfe_i32_zero_width_and_scc(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s1, 0x00000002",
            "s_bfe_i32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i32_zwid")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)

    def test_bfe_i32_clamped_width(self):
        # offset 28, width 8 -> clamped to 4 bits
        # 0x80000000 >> 28 = 0x8 (MSB=1 -> negative -8 -> 0xFFFFFFF8)
        lines = [
            "s_mov_b32 s0, 0x80000000",
            "s_mov_b32 s1, 0x0008001C",
            "s_bfe_i32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i32_clamp")
        self.assertEqual(words[0], 0xFFFFFFF8)

    def test_bfe_i32_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_bfe_i32_err", ["s_bfe_i32 s0", "s_endpgm"])

    def test_bfe_i32_mutation_detection(self):
        # Sabotage: if sign extension were missing, negative extract would yield positive 0x8
        lines = [
            "s_mov_b32 s0, 0x00000008",
            "s_mov_b32 s1, 0x00040000",
            "s_bfe_i32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i32_mut")
        self.assertNotEqual(words[0], 0x8)
        self.assertEqual(words[0], 0xFFFFFFF8)


# ==============================================================================
# 3. s_bfe_u64 Translator Tests
# ==============================================================================
class TestSBfeU64Translator(unittest.TestCase):
    def test_bfe_u64_cross_dword_extraction(self):
        # s[0:1] = (0x12345678 << 32) | 0xABCDEF01
        # offset 28, width 8: bits [35:28]
        # (0x12345678ABCDEF01 >> 28) & 0xFF = 0x8A
        lines = [
            "s_mov_b32 s0, 0xABCDEF01",
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x0008001C",
            "s_bfe_u64 s[4:5], s[0:1], s2",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u64_cross")
        self.assertEqual(words[0], 0x8A)
        self.assertEqual(words[1], 0)
        self.assertEqual(words[2], 1)

    def test_bfe_u64_zero_width_and_scc(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 0x00000010",  # width 0, offset 16
            "s_bfe_u64 s[4:5], s[0:1], s2",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u64_zwid")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)
        self.assertEqual(words[2], 0)

    def test_bfe_u64_high_dword_extraction(self):
        # Extract from high dword: offset 40, width 8
        # s[0:1] = (0x12345678 << 32) | 0x0
        # bits [47:40] of 0x1234567800000000 -> (0x12345678 >> 8) & 0xFF = 0x56
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_mov_b32 s1, 0x12345678",
            "s_mov_b32 s2, 0x00080028",  # width 8, offset 40
            "s_bfe_u64 s[4:5], s[0:1], s2",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u64_hi")
        self.assertEqual(words[0], 0x56)
        self.assertEqual(words[1], 0)

    def test_bfe_u64_clamped_boundary(self):
        # offset 60, width 16 -> clamped to 64 - 60 = 4 bits
        # s[0:1] = 0xF0000000_00000000 -> top 4 bits = 0xF
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_mov_b32 s1, 0xF0000000",
            "s_mov_b32 s2, 0x0010003C",  # width 16, offset 60
            "s_bfe_u64 s[4:5], s[0:1], s2",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u64_clamp")
        self.assertEqual(words[0], 0xF)
        self.assertEqual(words[1], 0)

    def test_bfe_u64_syntax_rejection(self):
        # Non-SGPR pair destination must be rejected
        with self.assertRaises(Exception):
            support.translate_lines("ps_bfe_u64_err", ["s_bfe_u64 s0, s[1:2], s3", "s_endpgm"])

    def test_bfe_u64_mutation_detection(self):
        lines = [
            "s_mov_b32 s0, 0x80",
            "s_mov_b32 s1, 0x00",
            "s_mov_b32 s2, 0x00080000",
            "s_bfe_u64 s[4:5], s[0:1], s2",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_u64_mut")
        self.assertEqual(words[0], 0x80)
        self.assertEqual(words[1], 0)


# ==============================================================================
# 4. s_bfe_i64 Translator Tests
# ==============================================================================
class TestSBfeI64Translator(unittest.TestCase):
    def test_bfe_i64_negative_sign_extension(self):
        # Extract 4 bits from bit 30 across dword boundary:
        # s[0] = 0x80000000, s[1] = 0x00000000
        # bits [33:30] of 0x0000000080000000 -> bit 31 is 1 -> value is 0x2
        # But if bit 33 were 1 -> negative. Let's make bit 33 = 1:
        # s[0] = 0x00000000, s[1] = 0x00000002 -> bit 33 is 1
        # offset 30, width 4: field extracted has MSB=1 -> negative
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_mov_b32 s1, 0x00000002",
            "s_mov_b32 s2, 0x0004001E",  # width 4, offset 30
            "s_bfe_i64 s[4:5], s[0:1], s2",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i64_neg")
        # Field is 0x8 (binary 1000) -> -8 in 4-bit signed -> sign-extended to 64-bit 0xFFFFFFFFFFFFFFF8
        self.assertEqual(words[0], 0xFFFFFFF8)
        self.assertEqual(words[1], 0xFFFFFFFF)
        self.assertEqual(words[2], 1)

    def test_bfe_i64_positive_sign_extension(self):
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_mov_b32 s1, 0x00000001",
            "s_mov_b32 s2, 0x0004001E",  # width 4, offset 30 -> field is 0x4 (MSB=0 -> positive 4)
            "s_bfe_i64 s[4:5], s[0:1], s2",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i64_pos")
        self.assertEqual(words[0], 4)
        self.assertEqual(words[1], 0)
        self.assertEqual(words[2], 1)

    def test_bfe_i64_zero_width_and_scc(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_mov_b32 s2, 0x00000020",  # width 0, offset 32
            "s_bfe_i64 s[4:5], s[0:1], s2",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i64_zwid")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0)
        self.assertEqual(words[2], 0)

    def test_bfe_i64_clamped_boundary(self):
        # offset 60, width 8 -> clamped to 4 bits
        # s[1] = 0x80000000 -> top 4 bits are 0x8 (MSB=1 -> negative -8)
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_mov_b32 s1, 0x80000000",
            "s_mov_b32 s2, 0x0008003C",
            "s_bfe_i64 s[4:5], s[0:1], s2",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i64_clamp")
        self.assertEqual(words[0], 0xFFFFFFF8)
        self.assertEqual(words[1], 0xFFFFFFFF)

    def test_bfe_i64_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_bfe_i64_err", ["s_bfe_i64 s[0:1], s[2:3]", "s_endpgm"])

    def test_bfe_i64_mutation_detection(self):
        # Mutating i64 to u64 would leave high dword as 0 instead of 0xFFFFFFFF
        lines = [
            "s_mov_b32 s0, 0x00000000",
            "s_mov_b32 s1, 0x00000002",
            "s_mov_b32 s2, 0x0004001E",
            "s_bfe_i64 s[4:5], s[0:1], s2",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfe_i64_mut")
        self.assertEqual(words[1], 0xFFFFFFFF)
        self.assertNotEqual(words[1], 0x00000000)


# ==============================================================================
# 5. s_bfm_b32 Translator Tests
# ==============================================================================
class TestSBfmB32Translator(unittest.TestCase):
    def test_bfm_b32_canonical(self):
        # width = 8, offset = 4 -> ((1 << 8) - 1) << 4 = 0xFF << 4 = 0xFF0
        lines = [
            "s_mov_b32 s0, 8",
            "s_mov_b32 s1, 4",
            "s_bfm_b32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b32_canon")
        self.assertEqual(words[0], 0xFF0)

    def test_bfm_b32_zero_width(self):
        # width = 0 -> (1 << 0) - 1 = 0
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 8",
            "s_bfm_b32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b32_zwid")
        self.assertEqual(words[0], 0)

    def test_bfm_b32_width_modulo_32(self):
        # width 32 mod 32 = 0 -> output 0
        # width 31, offset 0 -> 0x7FFFFFFF
        lines1 = [
            "s_mov_b32 s0, 32",
            "s_mov_b32 s1, 0",
            "s_bfm_b32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words1 = run_ps_words(lines1, "ps_bfm_b32_w32")
        self.assertEqual(words1[0], 0)

        lines2 = [
            "s_mov_b32 s0, 31",
            "s_mov_b32 s1, 0",
            "s_bfm_b32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words2 = run_ps_words(lines2, "ps_bfm_b32_w31")
        self.assertEqual(words2[0], 0x7FFFFFFF)

    def test_bfm_b32_offset_modulo_32(self):
        # offset 36 mod 32 = 4; width 4 -> 0xF << 4 = 0xF0
        lines = [
            "s_mov_b32 s0, 4",
            "s_mov_b32 s1, 36",
            "s_bfm_b32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b32_off36")
        self.assertEqual(words[0], 0xF0)

    def test_bfm_b32_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_bfm_b32_err", ["s_bfm_b32 s0, s1", "s_endpgm"])

    def test_bfm_b32_mutation_detection(self):
        lines = [
            "s_mov_b32 s0, 8",
            "s_mov_b32 s1, 4",
            "s_bfm_b32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b32_mut")
        self.assertNotEqual(words[0], 0xFF)


# ==============================================================================
# 6. s_bfm_b64 Translator Tests
# ==============================================================================
class TestSBfmB64Translator(unittest.TestCase):
    def test_bfm_b64_lane_active(self):
        # s_bfm_b64 produces a 64-bit lane mask.
        # When width = 16, offset = 0: lane 0 is in [0, 16) -> selected
        lines = [
            "s_mov_b32 s0, 16",
            "s_mov_b32 s1, 0",
            "s_bfm_b64 s[2:3], s0, s1",
            "v_cndmask_b32 v0, 0, 1, s[2:3]",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b64_on")
        self.assertEqual(words[0], 1)

    def test_bfm_b64_lane_inactive(self):
        # When width = 16, offset = 16: lane 0 is NOT in [16, 32) -> not selected (0)
        lines = [
            "s_mov_b32 s0, 16",
            "s_mov_b32 s1, 16",
            "s_bfm_b64 s[2:3], s0, s1",
            "v_cndmask_b32 v0, 0, 1, s[2:3]",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b64_off")
        self.assertEqual(words[0], 0)

    def test_bfm_b64_zero_width(self):
        # width = 0 -> no lanes active
        lines = [
            "s_mov_b32 s0, 0",
            "s_mov_b32 s1, 0",
            "s_bfm_b64 s[2:3], s0, s1",
            "v_cndmask_b32 v0, 0, 1, s[2:3]",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b64_zwid")
        self.assertEqual(words[0], 0)

    def test_bfm_b64_modulo_64(self):
        # width 64 mod 64 = 0 -> empty mask (0)
        lines1 = [
            "s_mov_b32 s0, 64",
            "s_mov_b32 s1, 0",
            "s_bfm_b64 s[2:3], s0, s1",
            "v_cndmask_b32 v0, 0, 1, s[2:3]",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words1 = run_ps_words(lines1, "ps_bfm_b64_w64")
        self.assertEqual(words1[0], 0)

        # width 16, offset 64 mod 64 = 0 -> lane 0 is in [0, 16) -> 1
        lines2 = [
            "s_mov_b32 s0, 16",
            "s_mov_b32 s1, 64",
            "s_bfm_b64 s[2:3], s0, s1",
            "v_cndmask_b32 v0, 0, 1, s[2:3]",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words2 = run_ps_words(lines2, "ps_bfm_b64_off64")
        self.assertEqual(words2[0], 1)

    def test_bfm_b64_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_bfm_b64_err", ["s_bfm_b64 s0, s1", "s_endpgm"])

    def test_bfm_b64_mutation_detection(self):
        # Sabotage check: offset 10 with width 5 should NOT activate lane 0
        lines = [
            "s_mov_b32 s0, 5",
            "s_mov_b32 s1, 10",
            "s_bfm_b64 s[2:3], s0, s1",
            "v_cndmask_b32 v0, 0, 1, s[2:3]",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0",
        ]
        words = run_ps_words(lines, "ps_bfm_b64_mut")
        self.assertNotEqual(words[0], 1)
        self.assertEqual(words[0], 0)


if __name__ == "__main__":
    unittest.main()
