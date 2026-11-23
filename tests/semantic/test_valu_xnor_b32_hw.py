from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import support

M = 0xFFFFFFFF


def ref_v_xnor_b32(s0: int, s1: int) -> int:
    """Mathematical reference oracle for v_xnor_b32: D = ~(S0 ^ S1) & 0xFFFFFFFF."""
    return (~((s0 & M) ^ (s1 & M))) & M


class TestValuXnorB32Hw(unittest.TestCase):
    """Translator-backed semantic tests for v_xnor_b32 on RDNA2 / SPIR-V."""

    def test_reference_oracle_identities(self):
        """Verify mathematical properties: identity with 0xFFFFFFFF, inversion with 0, self-equivalence, commutativity."""
        test_patterns = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0x80000000,
            0x7FFFFFFF,
            0xAAAAAAAA,
            0x55555555,
            0x0000FFFF,
            0xFFFF0000,
            0x0F0F0F0F,
        ]
        for val in test_patterns:
            with self.subTest(val=hex(val)):
                # Identity with all-ones: x XNOR 0xFFFFFFFF == x
                self.assertEqual(ref_v_xnor_b32(val, 0xFFFFFFFF), val)
                self.assertEqual(ref_v_xnor_b32(0xFFFFFFFF, val), val)

                # Inversion with zero: x XNOR 0 == ~x
                self.assertEqual(ref_v_xnor_b32(val, 0), (~val) & M)
                self.assertEqual(ref_v_xnor_b32(0, val), (~val) & M)

                # Self-equivalence: x XNOR x == 0xFFFFFFFF
                self.assertEqual(ref_v_xnor_b32(val, val), 0xFFFFFFFF)

                # Complementary: x XNOR (~x) == 0
                self.assertEqual(ref_v_xnor_b32(val, (~val) & M), 0x00000000)

        # Commutativity
        a, b = 0x12345678, 0x9ABCDEF0
        self.assertEqual(ref_v_xnor_b32(a, b), ref_v_xnor_b32(b, a))

    def test_bitwise_single_bit_walk(self):
        """Bitwise single-bit walks across all 32 bit positions."""
        for i in range(32):
            b0 = 1 << i
            b1 = 1 << ((i + 7) % 32)
            expected = (~(b0 ^ b1)) & M
            self.assertEqual(ref_v_xnor_b32(b0, b1), expected)

    def test_ieee754_float_bit_pattern_equivalence(self):
        """Bitwise equivalence tests across IEEE 754 representations."""
        # Comparing +0.0f (0x00000000) and -0.0f (0x80000000): differ only at sign bit
        self.assertEqual(ref_v_xnor_b32(0x00000000, 0x80000000), 0x7FFFFFFF)
        # Comparing +1.0f (0x3F800000) and -1.0f (0xBF800000): differ only at sign bit
        self.assertEqual(ref_v_xnor_b32(0x3F800000, 0xBF800000), 0x7FFFFFFF)
        # Identical quiet NaNs have matching bits
        qnan = 0x7FC00001
        self.assertEqual(ref_v_xnor_b32(qnan, qnan), 0xFFFFFFFF)

    def test_translator_literals_execution(self):
        """Translate v_xnor_b32 with inline literals and verify mrt0 output words."""
        lines = [
            "v_xnor_b32 v0, 0x12345678, 0x0000FFFF",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_xnor_literals", lines)
        self.assertIn("OpBitwiseXor", spv)
        self.assertIn("OpNot", spv)
        words = support.mrt0_words(spv)
        expected = ref_v_xnor_b32(0x12345678, 0x0000FFFF)
        self.assertEqual(words, [expected, expected, expected, expected])
        self.assertEqual(words[0], expected)

    def test_translator_sgpr_and_vgpr_operands(self):
        """Translate v_xnor_b32 with mixed SGPR and VGPR source operands."""
        s0_val = 0xAAAAAAAA
        v1_val = 0x55555555
        lines = [
            f"s_mov_b32 s0, {hex(s0_val)}",
            f"v_mov_b32 v1, {hex(v1_val)}",
            "v_xnor_b32 v0, s0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_xnor_sgpr_vgpr", lines)
        words = support.mrt0_words(spv)
        expected = ref_v_xnor_b32(s0_val, v1_val)
        self.assertEqual(words, [expected, expected, expected, expected])
        self.assertEqual(words[0], 0x00000000)

    def test_translator_register_aliasing_in_place(self):
        """In-place register aliasing where destination overwrites src0 or src1."""
        cases = [
            # dst == src0
            (
                [
                    "v_mov_b32 v0, 0x12345678",
                    "v_mov_b32 v1, 0x12340000",
                    "v_xnor_b32 v0, v0, v1",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ],
                ref_v_xnor_b32(0x12345678, 0x12340000),
            ),
            # dst == src1
            (
                [
                    "v_mov_b32 v0, 0x12345678",
                    "v_mov_b32 v1, 0x12340000",
                    "v_xnor_b32 v1, v0, v1",
                    "exp mrt0 v1, v1, v1, v1 done vm",
                    "s_endpgm",
                ],
                ref_v_xnor_b32(0x12345678, 0x12340000),
            ),
            # dst == src0 == src1
            (
                [
                    "v_mov_b32 v0, 0xDEADBEEF",
                    "v_xnor_b32 v0, v0, v0",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ],
                0xFFFFFFFF,
            ),
        ]
        for lines, expected in cases:
            spv = support.translate_lines("ps_xnor_aliasing", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words, [expected, expected, expected, expected])

    def test_translator_four_channel_divergent_mrt0(self):
        """Compute four distinct 32-bit channels with v_xnor_b32 into mrt0."""
        lines = [
            "v_xnor_b32 v0, 0x00000000, 0xFFFFFFFF",
            "v_xnor_b32 v1, 0xFFFFFFFF, 0xFFFFFFFF",
            "v_xnor_b32 v2, 0xAAAAAAAA, 0xAAAAAAAA",
            "v_xnor_b32 v3, 0x12345678, 0xEDCBA987",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_xnor_4channel", lines)
        words = support.mrt0_words(spv)
        exp0 = ref_v_xnor_b32(0x00000000, 0xFFFFFFFF)
        exp1 = ref_v_xnor_b32(0xFFFFFFFF, 0xFFFFFFFF)
        exp2 = ref_v_xnor_b32(0xAAAAAAAA, 0xAAAAAAAA)
        exp3 = ref_v_xnor_b32(0x12345678, 0xEDCBA987)
        self.assertEqual(words, [exp0, exp1, exp2, exp3])

    def test_negative_syntax_rejection(self):
        """Translator rejects malformed v_xnor_b32 instructions missing operands."""
        malformed = [
            "v_xnor_b32 v0, v1",  # missing src1
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_xnor_malformed", malformed)


if __name__ == "__main__":
    unittest.main()
