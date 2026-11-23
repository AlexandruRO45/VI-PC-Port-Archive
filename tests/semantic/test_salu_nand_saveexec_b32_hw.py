from __future__ import annotations

import enum
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

"""
RDNA2 SALU instruction semantic tests: s_nand_saveexec_b32.

Exercises s_nand_saveexec_b32 through tools/rdna2spirv.py and spirv_mini_interp.
RDNA2 ISA, SOP1:
    s_nand_saveexec_b32 sD, sS0
Semantics:
    D = EXEC_LO
    EXEC_LO = ~(S0 & EXEC_LO) & 0xFFFFFFFF
    SCC = (EXEC_LO != 0)

In Wave64 mode:
    EXEC_HI (lanes 32..63) remains untouched.
    D receives the 32-bit old EXEC_LO.
    SCC reflects (new EXEC_LO != 0).
"""

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def ref_nand_saveexec_b32(s0: int, old_exec_lo: int) -> Tuple[int, int, int]:
    """
    Reference mathematical model for s_nand_saveexec_b32.
    Returns (new_exec_lo, saved_d, scc).
    """
    s0_32 = s0 & MASK32
    exec_lo = old_exec_lo & MASK32
    new_exec_lo = (~(s0_32 & exec_lo)) & MASK32
    saved_d = exec_lo
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, saved_d, scc


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle
# ==============================================================================

class TestSaluNandSaveexecB32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical model of s_nand_saveexec_b32."""

    def test_both_zero(self) -> None:
        new_exec, saved_d, scc = ref_nand_saveexec_b32(0, 0)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(saved_d, 0)
        self.assertEqual(scc, 1)

    def test_all_ones_self_cancellation(self) -> None:
        new_exec, saved_d, scc = ref_nand_saveexec_b32(MASK32, MASK32)
        self.assertEqual(new_exec, 0)
        self.assertEqual(saved_d, MASK32)
        self.assertEqual(scc, 0)

    def test_orthogonal_masks(self) -> None:
        # S0 and EXEC have no overlapping bits -> S0 & EXEC == 0 -> NAND is ~0
        s0 = 0x5555_5555
        old_exec = 0xAAAA_AAAA
        new_exec, saved_d, scc = ref_nand_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(saved_d, old_exec)
        self.assertEqual(scc, 1)

    def test_partial_overlap(self) -> None:
        s0 = 0x0000_FFFF
        old_exec = 0x00FF_00FF
        # s0 & old_exec = 0x0000_00FF
        # ~(0x0000_00FF) = 0xFFFF_FF00
        new_exec, saved_d, scc = ref_nand_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0xFFFF_FF00)
        self.assertEqual(saved_d, old_exec)
        self.assertEqual(scc, 1)

    def test_single_bit_patterns(self) -> None:
        for bit in range(32):
            val = 1 << bit
            new_exec, saved_d, scc = ref_nand_saveexec_b32(val, val)
            self.assertEqual(new_exec, (~val) & MASK32)
            self.assertEqual(saved_d, val)
            self.assertEqual(scc, 1)

if __name__ == "__main__":
    unittest.main()
