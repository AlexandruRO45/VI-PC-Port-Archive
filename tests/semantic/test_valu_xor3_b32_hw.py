from __future__ import annotations

"""Hardware and translator semantic tests for RDNA2 VALU v_xor3_b32 through tools/rdna2spirv.py.

Instruction:
  v_xor3_b32 D, S0, S1, S2  (VOP3 format, 32-bit vector bitwise 3-input XOR)

Architectural Semantics:
  D = (S0 ^ S1 ^ S2) & 0xFFFFFFFF

Mathematical Properties:
  - Three-operand bitwise XOR (parity function across three 32-bit inputs).
  - Associative and fully symmetric:
      v_xor3_b32(A, B, C) == v_xor3_b32(B, C, A) == v_xor3_b32(C, A, B)
  - Self-identity under triple occurrence:
      v_xor3_b32(A, A, A) == A
  - Pair cancellation:
      v_xor3_b32(A, A, B) == B
      v_xor3_b32(A, B, A) == B
      v_xor3_b32(B, A, A) == B
  - Inversion:
      v_xor3_b32(A, 0xFFFFFFFF, 0) == (~A) & 0xFFFFFFFF
  - Parity truth table:
      Bit k of D is 1 iff an odd number (1 or 3) of {S0[k], S1[k], S2[k]} are 1.

Translation Model:
  rdna2spirv translates v_xor3_b32 into two chained OpBitwiseXor instructions:
    %s = OpBitwiseXor %uint %x %y
    %r = OpBitwiseXor %uint %s %z
    dst = %r
  Verified against SPIR-V execution via spirv_mini_interp and support.mrt0_words.
"""

import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # Reaches tools/rdna2spirv.py

MASK32: int = 0xFFFF_FFFF


def ref_xor3_b32(s0: int, s1: int, s2: int) -> int:
    """Pure reference mathematical oracle for 32-bit 3-input XOR."""
    return ((s0 & MASK32) ^ (s1 & MASK32) ^ (s2 & MASK32)) & MASK32


def run_xor3_shader(s0: int, s1: int, s2: int) -> int:
    """Translate and execute a synthetic pixel shader evaluating v_xor3_b32."""
    lines = [
        f"v_mov_b32 v0, 0x{s0 & MASK32:08x}",
        f"v_mov_b32 v1, 0x{s1 & MASK32:08x}",
        f"v_mov_b32 v2, 0x{s2 & MASK32:08x}",
        "v_xor3_b32 v3, v0, v1, v2",
        "exp mrt0 v3, v3, v3, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_xor3_run", lines)
    words = support.mrt0_words(spv)
    return words[0]


class TestValuXor3B32Translator(unittest.TestCase):
    """Semantic tests for v_xor3_b32 through the rdna2spirv translator and interpreter."""

    def test_spirv_emission_structure(self) -> None:
        """Verify that translate_lines produces two chained OpBitwiseXor instructions."""
        lines = [
            "v_mov_b32 v0, 0x11111111",
            "v_mov_b32 v1, 0x22222222",
            "v_mov_b32 v2, 0x44444444",
            "v_xor3_b32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_xor3_emit", lines)
        self.assertGreaterEqual(spv.count("OpBitwiseXor"), 2)

    def test_basic_truth_table_patterns(self) -> None:
        """Test truth table across distinct bit patterns."""
        cases = [
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0x00000000, 0x00000000, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xAAAAAAAA, 0x55555555, 0x00000000, 0xFFFFFFFF),
            (0xAAAAAAAA, 0x55555555, 0xFFFFFFFF, 0x00000000),
            (0x12345678, 0x9ABCDEF0, 0x0F0F0F0F, 0x12345678 ^ 0x9ABCDEF0 ^ 0x0F0F0F0F),
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                ref = ref_xor3_b32(s0, s1, s2)
                self.assertEqual(ref, expected)
                got = run_xor3_shader(s0, s1, s2)
                self.assertEqual(got, expected)

    def test_symmetry_and_permutation_invariance(self) -> None:
        """v_xor3_b32 is invariant under all 6 permutations of its three arguments."""
        a, b, c = 0xDEADBEEF, 0xCAFEBABE, 0x01234567
        expected = ref_xor3_b32(a, b, c)

        permutations = [
            (a, b, c),
            (a, c, b),
            (b, a, c),
            (b, c, a),
            (c, a, b),
            (c, b, a),
        ]
        for p0, p1, p2 in permutations:
            got = run_xor3_shader(p0, p1, p2)
            self.assertEqual(got, expected)

    def test_pair_cancellation_identity(self) -> None:
        """Any duplicate pair cancels out: xor3(A, A, B) == B."""
        patterns = [0x00000000, 0x12345678, 0x80000000, 0xFFFFFFFF]
        test_val = 0xA5A5A5A5
        for p in patterns:
            self.assertEqual(run_xor3_shader(p, p, test_val), test_val)
            self.assertEqual(run_xor3_shader(p, test_val, p), test_val)
            self.assertEqual(run_xor3_shader(test_val, p, p), test_val)

    def test_triple_self_identity(self) -> None:
        """Triple occurrence yields the identity: xor3(A, A, A) == A."""
        patterns = [0x00000000, 0x1, 0xCAFE0000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
        for p in patterns:
            self.assertEqual(run_xor3_shader(p, p, p), p)

    def test_bitwise_not_inversion(self) -> None:
        """xor3(A, ~0, 0) computes bitwise NOT of A."""
        values = [0x00000000, 0x0000FFFF, 0x12345678, 0xAAAAAAAA, 0xFFFFFFFF]
        for v in values:
            expected = (~v) & MASK32
            got = run_xor3_shader(v, 0xFFFFFFFF, 0)
            self.assertEqual(got, expected)

    def test_sgpr_and_inline_constants_mixed(self) -> None:
        """Verify v_xor3_b32 with SGPR uniform operand and inline constants."""
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "v_mov_b32 v0, 0x00000001",
            "v_xor3_b32 v1, s0, v0, 0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_xor3_sgpr_inline", lines)
        words = support.mrt0_words(spv)
        expected = 0x12345678 ^ 0x00000001 ^ 0
        self.assertEqual(words[0], expected)

    def test_negative_controls(self) -> None:
        """Distinguish XOR3 from OR3, ADD3, and standard 2-input XOR."""
        a, b, c = 0x1, 0x1, 0x1
        xor3_res = ref_xor3_b32(a, b, c)
        add3_res = (a + b + c) & MASK32
        xor2_res = (a ^ b) & MASK32

        self.assertNotEqual(xor3_res, add3_res)
        self.assertNotEqual(xor3_res, xor2_res)

        a2, b2, c2 = 0x1, 0x1, 0x0
        self.assertEqual(ref_xor3_b32(a2, b2, c2), 0)
        self.assertEqual((a2 | b2 | c2) & MASK32, 1)

    def test_randomized_vectors_against_reference(self) -> None:
        """Verify pseudorandom vector triples through translator against reference oracle."""
        rng = random.Random(0x4C454F4E)
        for _ in range(12):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            s2 = rng.getrandbits(32)
            expected = ref_xor3_b32(s0, s1, s2)
            got = run_xor3_shader(s0, s1, s2)
            self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
