"""Tests for tools/rdna2spirv.py same_encoding() matching logic (BUILD-250)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from rdna2spirv import same_encoding


class TestSameEncoding(unittest.TestCase):
    def setUp(self):
        # Sample plain-VOP3 instruction: v_fma_f16 v0, v1, v2, v3
        # byte 3 is 0xd7 -> (0xd7 >> 2) == 0x35 (VOP3)
        self.plain_vop3 = bytes([0x00, 0x00, 0x4B, 0xD7, 0x01, 0x05, 0x0E, 0x04])

        # Sample VOP3P instruction: v_pk_add_f16 v0, v1, v2
        # byte 3 is 0xCC (VOP3P)
        self.vop3p_twosrc = bytes([0x00, 0x00, 0x00, 0xCC, 0x01, 0x05, 0x80, 0x00])

        # Non-VOP3 instruction: byte 3 has different high bits
        self.non_vop3 = bytes([0x00, 0x00, 0x00, 0x7E, 0x01, 0x05, 0x02, 0x00])

    def test_exact_match(self):
        self.assertTrue(same_encoding(self.plain_vop3, self.plain_vop3, "v_fma_f16"))
        self.assertTrue(same_encoding(self.vop3p_twosrc, self.vop3p_twosrc, "v_pk_add_f16"))

    def test_invalid_lengths(self):
        self.assertFalse(same_encoding(self.plain_vop3, self.plain_vop3[:7], "v_fma_f16"))
        self.assertFalse(same_encoding(b"\x00" * 4, b"\x01" * 4, "v_nop"))

    def test_plain_vop3_op_sel_hi2_tolerated(self):
        # BUILD-250: LLVM 23.1.5 re-encodes plain-VOP3 src2 op_sel_hi[2] (dword1 bit 25 = 64-bit bit 57).
        # In little-endian, bit 57 is bit 1 of byte 7.
        raw = bytearray(self.plain_vop3)
        printed = bytearray(self.plain_vop3)
        printed[7] ^= (1 << 1)  # toggle bit 57 (dword1 bit 25)

        self.assertTrue(same_encoding(bytes(raw), bytes(printed), "v_fma_f16"))
        self.assertTrue(same_encoding(bytes(printed), bytes(raw), "v_fma_f16"))

    def test_plain_vop3_other_bit_differences_rejected(self):
        # Different opcode or operand bit must fail
        raw = bytearray(self.plain_vop3)
        printed = bytearray(self.plain_vop3)
        printed[0] ^= 0x01  # change destination register
        self.assertFalse(same_encoding(bytes(raw), bytes(printed), "v_fma_f16"))

        # Different bit in byte 7 (e.g. bit 0 of byte 7 = bit 56) must fail
        printed2 = bytearray(self.plain_vop3)
        printed2[7] ^= (1 << 0)
        self.assertFalse(same_encoding(bytes(raw), bytes(printed2), "v_fma_f16"))

    def test_non_vop3_bit57_rejected(self):
        raw = bytearray(self.non_vop3)
        printed = bytearray(self.non_vop3)
        printed[7] ^= (1 << 1)
        self.assertFalse(same_encoding(bytes(raw), bytes(printed), "v_nop"))

    def test_vop3p_twosrc_bit14_tolerated(self):
        # Existing rule: VOP3P bit 14 tolerated for two-source packed ops
        raw = bytearray(self.vop3p_twosrc)
        printed = bytearray(self.vop3p_twosrc)
        # bit 14 is in byte 1 (bit 6 of byte 1: 8 + 6 = 14)
        printed[1] ^= (1 << 6)
        self.assertTrue(same_encoding(bytes(raw), bytes(printed), "v_pk_add_f16"))

    def test_vop3p_three_source_bit14_rejected(self):
        raw = bytearray(self.vop3p_twosrc)
        printed = bytearray(self.vop3p_twosrc)
        printed[1] ^= (1 << 6)
        # Three-source packed op should not forgive bit 14
        self.assertFalse(same_encoding(bytes(raw), bytes(printed), "v_pk_fma_f16"))

    def test_unused_src_fields_tolerated(self):
        # Unused SRC2 field (bits 58:50): stored as 0 in raw, printed as 0x80
        # When r has 0 in bits 58:50, and printed has 0x80:
        base = bytearray([0x00, 0x00, 0x32, 0xD5, 0x01, 0x05, 0x00, 0x00])  # v_mul_f32_e64
        # Set bits 58:50 in printed to 0x80:
        val = int.from_bytes(base, "little")
        val |= (0x80 << 50)
        printed = val.to_bytes(8, "little")
        self.assertTrue(same_encoding(bytes(base), printed, "v_mul_legacy_f32"))


if __name__ == "__main__":
    unittest.main()
