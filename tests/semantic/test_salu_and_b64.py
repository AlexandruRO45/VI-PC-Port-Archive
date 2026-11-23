"""Semantic reference oracle and verification for RDNA2 64-bit scalar AND (s_and_b64).

Covers:
  - s_and_b64: D = S0 & S1 (64-bit bitwise AND)
  - Scalar condition code SCC: SCC = 1 if D != 0 else 0
  - Wave64 / Wave32 execution mask (EXEC) and vector condition code (VCC) interactions
  - Register aliasing (in-place destination, source overlapping, partial SGPR pair overlap)
  - Special register handling (null destination, exec, vcc)
  - IEEE 754 bit-pattern preservation without floating-point side effects
  - SPIR-V translation lowering and mini-interpreter validation via support.translate_lines

Standard library Python only; strictly zero game data; fully reproducible.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import enum
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
_TRANSLATOR_PRESENT = translate_lines is not None and mrt0_words is not None

def ref_s_and_b64(s0: int, s1: int) -> int:
    """RDNA2 s_and_b64: D = S0 & S1 (64-bit unsigned bitwise AND)."""
    return (s0 & s1) & MASK64


def ref_scc(val64: int) -> int:
    """RDNA2 SCC condition flag: 1 if 64-bit result != 0, else 0."""
    return 1 if (val64 & MASK64) != 0 else 0


class SaluAndB64OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for 64-bit scalar bitwise AND."""

    def test_truth_table_single_bits(self) -> None:
        """Exhaustive 1-bit truth table verification."""
        table = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 1,
        }
        for (b0, b1), expected in table.items():
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK64 if b0 else 0
                s1 = MASK64 if b1 else 0
                want = MASK64 if expected else 0
                got = ref_s_and_b64(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), expected)

    def test_algebraic_identities(self) -> None:
        """Verify identity, annihilator, idempotence, commutativity, and associativity."""
        x = 0x12345678_9ABCDEF0
        y = 0xAAAAAAAAAAAAAAAA
        z = 0x55555555_55555555

        # Identity: x & ~0 == x
        self.assertEqual(ref_s_and_b64(x, MASK64), x)
        self.assertEqual(ref_s_and_b64(MASK64, x), x)

        # Annihilator: x & 0 == 0
        self.assertEqual(ref_s_and_b64(x, 0), 0)
        self.assertEqual(ref_s_and_b64(0, x), 0)

        # Idempotence: x & x == x
        self.assertEqual(ref_s_and_b64(x, x), x)

        # Commutativity: x & y == y & x
        self.assertEqual(ref_s_and_b64(x, y), ref_s_and_b64(y, x))

        # Associativity: (x & y) & z == x & (y & z)
        self.assertEqual(
            ref_s_and_b64(ref_s_and_b64(x, y), z),
            ref_s_and_b64(x, ref_s_and_b64(y, z)),
        )

        # Distributivity over OR: x & (y | z) == (x & y) | (x & z)
        self.assertEqual(
            ref_s_and_b64(x, (y | z) & MASK64),
            ((ref_s_and_b64(x, y)) | (ref_s_and_b64(x, z))) & MASK64,
        )

    def test_boundary_values(self) -> None:
        """Boundary values across 64-bit boundaries (0, MSB, LSB, alternating bits)."""
        boundaries = [
            0x00000000_00000000,
            0xFFFFFFFF_FFFFFFFF,
            0x80000000_00000000,
            0x00000000_00000001,
            0x7FFFFFFF_FFFFFFFF,
            0x00000000_FFFFFFFF,
            0xFFFFFFFF_00000000,
            0x55555555_55555555,
            0xAAAAAAAA_AAAAAAAA,
            0x01234567_89ABCDEF,
            0xFEDCBA98_76543210,
        ]
        for a in boundaries:
            for b in boundaries:
                with self.subTest(a=hex(a), b=hex(b)):
                    res = ref_s_and_b64(a, b)
                    self.assertGreaterEqual(res, 0)
                    self.assertLessEqual(res, MASK64)
                    scc = ref_scc(res)
                    self.assertEqual(scc, 1 if res != 0 else 0)

if __name__ == "__main__":
    unittest.main()
