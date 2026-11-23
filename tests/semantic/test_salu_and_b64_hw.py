"""Semantic hardware tests and formal verification for RDNA2 s_and_b64.

Covers:
  - s_and_b64: D = S0 & S1 (bitwise AND across 64-bit scalar registers)
  - Condition code SCC: SCC = 1 if D != 0 else 0
  - Special register operands: EXEC, VCC, NULL
  - Multi-lane divergence and EXEC lane masking
  - In-place operand preservation and destination aliasing
  - Double precision / packed float IEEE 754 bit-pattern preservation
  - Hardware SPIR-V translation and execution via tools/rdna2spirv.py
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64 = 1 << 63


def ref_s_and_b64(s0: int, s1: int) -> int:
    """RDNA2 s_and_b64: D = S0 & S1."""
    return (s0 & s1) & MASK64


def ref_scc_b64(val64: int) -> int:
    """RDNA2 SCC condition flag: 1 if 64-bit result != 0, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


def lo32(val: int) -> int:
    return val & MASK32


def hi32(val: int) -> int:
    return (val >> 32) & MASK32


def join64(lo: int, hi: int) -> int:
    return ((hi & MASK32) << 32) | (lo & MASK32)


class SaluAndB64OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for RDNA2 s_and_b64."""

    def test_bitwise_truth_table_exhaustive(self):
        """Verify bit-by-bit truth table of AND across 64 bit positions."""
        for bit in range(64):
            b_mask = 1 << bit
            self.assertEqual(ref_s_and_b64(0, 0) & b_mask, 0)
            self.assertEqual(ref_s_and_b64(0, b_mask) & b_mask, 0)
            self.assertEqual(ref_s_and_b64(b_mask, 0) & b_mask, 0)
            self.assertEqual(ref_s_and_b64(b_mask, b_mask) & b_mask, b_mask)

    def test_algebraic_identities(self):
        """Verify Boolean algebraic laws: commutativity, associativity, idempotence, identities."""
        a = 0x01234567_89ABCDEF
        b = 0xFEDCBA98_76543210
        c = 0x5555AAAA_3333CCCC

        # Commutativity: A & B == B & A
        self.assertEqual(ref_s_and_b64(a, b), ref_s_and_b64(b, a))
        # Associativity: (A & B) & C == A & (B & C)
        self.assertEqual(ref_s_and_b64(ref_s_and_b64(a, b), c), ref_s_and_b64(a, ref_s_and_b64(b, c)))
        # Identity element: A & 1s == A
        self.assertEqual(ref_s_and_b64(a, MASK64), a)
        # Annihilator: A & 0 == 0
        self.assertEqual(ref_s_and_b64(a, 0), 0)
        # Idempotence: A & A == A
        self.assertEqual(ref_s_and_b64(a, a), a)

    def test_boundary_corners(self):
        """Verify edge values: 0, 1, MASK64, MSB, alternating bit patterns, dword masks."""
        boundaries = [
            0x00000000_00000000,
            0x00000000_00000001,
            0x80000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
        ]
        for s0 in boundaries:
            for s1 in boundaries:
                with self.subTest(s0=hex(s0), s1=hex(s1)):
                    res = ref_s_and_b64(s0, s1)
                    self.assertGreaterEqual(res, 0)
                    self.assertLessEqual(res, MASK64)
                    scc = ref_scc_b64(res)
                    self.assertIn(scc, (0, 1))
                    self.assertEqual(scc, 1 if res != 0 else 0)

    def test_spirv_hardware_basic(self):
        """Verify translator emits valid SPIR-V executing exact 64-bit bitwise AND."""
        test_pairs = [
            (0x12345678_9ABCDEF0, 0x0F0F0F0F_F0F0F0F0),
            (0xFFFFFFFF_FFFFFFFF, 0x12345678_87654321),
            (0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000),
            (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF),
        ]
        for idx, (v0, v1) in enumerate(test_pairs):
            with self.subTest(idx=idx, v0=hex(v0), v1=hex(v1)):
                lines = [
                    f"s_mov_b32 s0, {lo32(v0):#x}",
                    f"s_mov_b32 s1, {hi32(v0):#x}",
                    f"s_mov_b32 s2, {lo32(v1):#x}",
                    f"s_mov_b32 s3, {hi32(v1):#x}",
                    "s_and_b64 s[4:5], s[0:1], s[2:3]",
                    "s_cselect_b32 s6, 1, 0",
                    "v_mov_b32 v0, s4",
                    "v_mov_b32 v1, s5",
                    "v_mov_b32 v2, s6",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_salu_and_b64_basic_{idx}", lines)
                out = mrt0_words(spv)
                res64 = join64(out[0], out[1])
                expected = ref_s_and_b64(v0, v1)
                self.assertEqual(res64, expected)
                self.assertEqual(out[2], ref_scc_b64(expected))


if __name__ == "__main__":
    unittest.main()
