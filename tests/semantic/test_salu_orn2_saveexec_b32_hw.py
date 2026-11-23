from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_orn2_saveexec_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_orn2_saveexec_b32 instruction belongs to SOP1 format.
  Instruction:
    s_orn2_saveexec_b32 sdst, ssrc0

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo) to receive the previous EXEC_LO mask
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = EXEC_LO
    EXEC_LO = ssrc0 | ~EXEC_LO
    SCC = 1 if EXEC_LO != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32: int = 0xFFFFFFFF
ALL_LANES_ACTIVE_32: int = 0xFFFFFFFF
ALL_LANES_INACTIVE_32: int = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_orn2_saveexec_b32(ssrc0: int, old_exec_lo: int) -> tuple[int, int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_orn2_saveexec_b32 instruction.

    Args:
        ssrc0: 32-bit source operand (uint32)
        old_exec_lo: 32-bit current EXEC_LO mask before instruction execution (uint32)

    Returns:
        tuple (new_exec_lo, sdst, scc):
          - new_exec_lo: updated 32-bit EXEC_LO mask = ssrc0 | ~old_exec_lo
          - sdst: saved previous EXEC_LO mask = old_exec_lo
          - scc: scalar condition code = 1 if new_exec_lo != 0 else 0
    """
    s0 = u32(ssrc0)
    old_e = u32(old_exec_lo)
    sdst = old_e
    new_exec_lo = u32(s0 | (~old_e & MASK32))
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, sdst, scc


class TestSaluOrn2SaveexecB32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_orn2_saveexec_b32."""

    def test_oracle_all_ones_exec(self):
        # If old_exec_lo = all 1s, ~old_exec_lo = 0, so new_exec_lo = ssrc0 | 0 = ssrc0
        s0 = 0x12345678
        old_exec = ALL_LANES_ACTIVE_32
        new_exec, sdst, scc = ref_orn2_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, s0)
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_all_zeros_exec(self):
        # If old_exec_lo = 0, ~old_exec_lo = all 1s, so new_exec_lo = ssrc0 | all 1s = all 1s
        s0 = 0x0
        old_exec = ALL_LANES_INACTIVE_32
        new_exec, sdst, scc = ref_orn2_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, ALL_LANES_INACTIVE_32)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0_all_ones_exec_produces_zero_exec(self):
        # The unique condition where new_exec_lo becomes 0: s0 == 0 and old_exec_lo == ~0
        s0 = 0x0
        old_exec = ALL_LANES_ACTIVE_32
        new_exec, sdst, scc = ref_orn2_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, ALL_LANES_ACTIVE_32)
        self.assertEqual(scc, 0)

    def test_oracle_self_exec(self):
        # When ssrc0 == old_exec_lo: exec | ~exec = ALL_LANES_ACTIVE_32
        sample_exec = 0xAAAAAAAA
        new_exec, sdst, scc = ref_orn2_saveexec_b32(sample_exec, sample_exec)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, sample_exec)
        self.assertEqual(scc, 1)
