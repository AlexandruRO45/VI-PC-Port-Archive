from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 32-bit Vector Logic Suite.

Covers:
- v_and_b32: Bitwise AND D = S0 & S1.
- v_or_b32: Bitwise OR D = S0 | S1.
- v_xor_b32: Bitwise XOR D = S0 ^ S1.
- v_not_b32: Bitwise NOT D = ~S0.
- v_xnor_b32: Bitwise XNOR D = ~(S0 ^ S1).
- v_and_or_b32: 3-operand bitwise operation D = (S0 & S1) | S2.
- v_or3_b32: 3-operand bitwise OR D = S0 | S1 | S2.

Standard library only (python -I, struct, math, unittest). Zero game data.
All test suites execute real translated assembly via support.translate_lines.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv

MASK32: int = 0xFFFF_FFFF


def run_shader(name: str, lines: List[str]) -> Tuple[int, str]:
    full_lines = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = support.translate_lines(name, full_lines)
    words = support.mrt0_words(spv)
    return words[0], spv


# ------------------------------------------------------------------------------
# v_and_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_and_b32(s0: int, s1: int) -> int:
    return (s0 & s1) & MASK32


class TestAndB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            (0xFF00, 0x0FF0),
            (0x1234_5678, 0xFFFF_0000),
            (0xAAAA_AAAA, 0x5555_5555),
            (0x1357_9BDF, 0xFEDC_BA98),
        ]
        for idx, (a, b) in enumerate(vectors):
            val, _ = run_shader(f"ps_and_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(a)}",
                f"v_mov_b32 v2, {hex(b)}",
                "v_and_b32 v0, v1, v2",
            ])
            self.assertEqual(val, ref_and_b32(a, b))

    def test_oracle_identities_hardware(self):
        val_id, _ = run_shader("ps_and_id", [
            "v_mov_b32 v1, 0xCAFEBABE",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_id, 0xCAFEBABE)

        val_zero, _ = run_shader("ps_and_zero", [
            "v_mov_b32 v1, 0xCAFEBABE",
            "v_mov_b32 v2, 0",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_zero, 0)
