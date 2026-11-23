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


def ref_v_or3_b32(s0: int, s1: int, s2: int) -> int:
    """Mathematical reference oracle for v_or3_b32: D = (S0 | S1 | S2) & 0xFFFFFFFF."""
    return ((s0 & M) | (s1 & M) | (s2 & M)) & M


class TestValuOr3B32Hw(unittest.TestCase):
    """Translator-backed semantic tests for v_or3_b32 on RDNA2 / SPIR-V."""

    def test_reference_oracle_identities(self):
        """Verify mathematical properties: zero identity, all-ones absorbing, idempotency, commutativity."""
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
                # Identity with zeros
                self.assertEqual(ref_v_or3_b32(val, 0, 0), val)
                self.assertEqual(ref_v_or3_b32(0, val, 0), val)
                self.assertEqual(ref_v_or3_b32(0, 0, val), val)

                # Absorbing with all-ones
                self.assertEqual(ref_v_or3_b32(val, 0xFFFFFFFF, 0), 0xFFFFFFFF)
                self.assertEqual(ref_v_or3_b32(0, 0, 0xFFFFFFFF), 0xFFFFFFFF)

                # Idempotency
                self.assertEqual(ref_v_or3_b32(val, val, val), val)

        # Commutativity across permutations
        a, b, c = 0x11223344, 0x55667788, 0x99AABBCC
        res = ref_v_or3_b32(a, b, c)
        for perm in [
            (a, b, c),
            (a, c, b),
            (b, a, c),
            (b, c, a),
            (c, a, b),
            (c, b, a),
        ]:
            self.assertEqual(ref_v_or3_b32(*perm), res)

    def test_bitwise_disjoint_fields_and_single_bit_walk(self):
        """Disjoint bitfields combination and 32-bit single-bit walks."""
        # 3 disjoint partitions summing to all-ones
        p0 = 0xFF000000
        p1 = 0x00FF0000
        p2 = 0x0000FFFF
        self.assertEqual(ref_v_or3_b32(p0, p1, p2), 0xFFFFFFFF)

        # Single-bit walk across 32 bit positions
        for i in range(32):
            b0 = 1 << i
            b1 = 1 << ((i + 11) % 32)
            b2 = 1 << ((i + 23) % 32)
            expected = b0 | b1 | b2
            self.assertEqual(ref_v_or3_b32(b0, b1, b2), expected)

    def test_ieee754_float_bit_pattern_assembly(self):
        """Assembling sign, exponent, and mantissa bitfields into IEEE 754 representations."""
        # Assembly of +1.0f: sign=0, exp=0x3F800000, mantissa=0
        self.assertEqual(ref_v_or3_b32(0, 0x3F800000, 0), 0x3F800000)
        # Assembly of -1.0f: sign=0x80000000, exp=0x3F800000, mantissa=0
        self.assertEqual(ref_v_or3_b32(0x80000000, 0x3F800000, 0), 0xBF800000)
        # Assembly of NaN / Infinity / Subnormal bit patterns
        sign_bit = 0x80000000
        inf_exp = 0x7F800000
        nan_payload = 0x00400001
        qnan = ref_v_or3_b32(sign_bit, inf_exp, nan_payload)
        self.assertEqual(qnan, 0xFFC00001)

    def test_translator_literals_execution(self):
        """Translate v_or3_b32 with inline literals and verify mrt0 output words."""
        lines = [
            "v_or3_b32 v0, 0x12000000, 0x00340000, 0x00005678",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_or3_literals", lines)
        self.assertIn("OpBitwiseOr", spv)
        words = support.mrt0_words(spv)
        expected = ref_v_or3_b32(0x12000000, 0x00340000, 0x00005678)
        self.assertEqual(words, [expected, expected, expected, expected])
        self.assertEqual(words[0], 0x12345678)

    def test_translator_sgpr_and_vgpr_operands(self):
        """Translate v_or3_b32 with mixed SGPR and VGPR source operands."""
        s0_val = 0xAA000000
        v1_val = 0x00BB0000
        v2_val = 0x0000CCDD
        lines = [
            f"s_mov_b32 s0, {hex(s0_val)}",
            f"v_mov_b32 v1, {hex(v1_val)}",
            f"v_mov_b32 v2, {hex(v2_val)}",
            "v_or3_b32 v0, s0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_or3_sgpr_vgpr", lines)
        words = support.mrt0_words(spv)
        expected = ref_v_or3_b32(s0_val, v1_val, v2_val)
        self.assertEqual(words, [expected, expected, expected, expected])
        self.assertEqual(words[0], 0xAABBCCDD)

    def test_translator_register_aliasing_in_place(self):
        """In-place register aliasing where destination overwrites src0, src1, or src2."""
        cases = [
            # dst == src0
            (
                [
                    "v_mov_b32 v0, 0x10000000",
                    "v_mov_b32 v1, 0x02000000",
                    "v_mov_b32 v2, 0x00300000",
                    "v_or3_b32 v0, v0, v1, v2",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ],
                0x12300000,
            ),
            # dst == src1
            (
                [
                    "v_mov_b32 v0, 0x10000000",
                    "v_mov_b32 v1, 0x02000000",
                    "v_mov_b32 v2, 0x00300000",
                    "v_or3_b32 v1, v0, v1, v2",
                    "exp mrt0 v1, v1, v1, v1 done vm",
                    "s_endpgm",
                ],
                0x12300000,
            ),
            # dst == src2
            (
                [
                    "v_mov_b32 v0, 0x10000000",
                    "v_mov_b32 v1, 0x02000000",
                    "v_mov_b32 v2, 0x00300000",
                    "v_or3_b32 v2, v0, v1, v2",
                    "exp mrt0 v2, v2, v2, v2 done vm",
                    "s_endpgm",
                ],
                0x12300000,
            ),
            # dst == src0 == src1 == src2
            (
                [
                    "v_mov_b32 v0, 0x77665544",
                    "v_or3_b32 v0, v0, v0, v0",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ],
                0x77665544,
            ),
        ]
        for lines, expected in cases:
            spv = support.translate_lines("ps_or3_aliasing", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words, [expected, expected, expected, expected])

    def test_translator_four_channel_divergent_mrt0(self):
        """Compute four distinct 32-bit channels with v_or3_b32 into mrt0."""
        lines = [
            "v_or3_b32 v0, 0x01000000, 0x00200000, 0x00000300",
            "v_or3_b32 v1, 0x10000000, 0x00040000, 0x00000008",
            "v_or3_b32 v2, 0xFF000000, 0x00FF0000, 0x000000FF",
            "v_or3_b32 v3, 0x55550000, 0x00005555, 0x00550000",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_or3_4channel", lines)
        words = support.mrt0_words(spv)
        exp0 = ref_v_or3_b32(0x01000000, 0x00200000, 0x00000300)
        exp1 = ref_v_or3_b32(0x10000000, 0x00040000, 0x00000008)
        exp2 = ref_v_or3_b32(0xFF000000, 0x00FF0000, 0x000000FF)
        exp3 = ref_v_or3_b32(0x55550000, 0x00005555, 0x00550000)
        self.assertEqual(words, [exp0, exp1, exp2, exp3])

    def test_negative_syntax_rejection(self):
        """Translator rejects malformed v_or3_b32 instructions missing operands."""
        malformed = [
            "v_or3_b32 v0, v1, v2",  # missing src2
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_or3_malformed", malformed)


if __name__ == "__main__":
    unittest.main()
