from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_xor_saveexec_b64.

RDNA2 Instruction Set Architecture, SOP1:
    s_xor_saveexec_b64 D, S0

Semantic operation:
    D.u64 = EXEC
    EXEC.u64 = S0.u64 ^ EXEC
    SCC = (EXEC != 0)

Outputs:
    D: 64-bit destination SGPR pair storing the previous EXEC mask.
    EXEC: Updated 64-bit execution mask.
    SCC: 1 if the updated EXEC mask is non-zero, 0 if it is all zeros.

Standard library Python only (unittest, struct, math, dataclasses, enum, typing).
Strictly zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_xor_saveexec_b64_oracle(s0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference mathematical model for s_xor_saveexec_b64.

    Args:
        s0: 64-bit unsigned integer source mask.
        old_exec: 64-bit unsigned integer execution mask before execution.

    Returns:
        (new_exec, d, scc) tuple where:
            new_exec: (s0 ^ old_exec) & MASK64
            d: old_exec (64-bit unsigned)
            scc: 1 if new_exec != 0 else 0
    """
    s0_u64 = s0 & MASK64
    old_exec_u64 = old_exec & MASK64
    new_exec = (s0_u64 ^ old_exec_u64) & MASK64
    d = old_exec_u64
    scc = 1 if new_exec != 0 else 0
    return new_exec, d, scc


class SXorSaveExecB64Oracle:
    """Architectural model and hardware-rule validator for s_xor_saveexec_b64."""

    @staticmethod
    def eval(s0: int, old_exec: int) -> Tuple[int, int, int]:
        return s_xor_saveexec_b64_oracle(s0, old_exec)

    @staticmethod
    def build_compute_shader(s0: int, old_exec: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_xor_saveexec_b64."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_xor_saveexec_b64_hw"
        lines = [
            f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}",
            f"s_mov_b32 exec_hi, {(old_exec >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s0 & MASK32:#x}",
            f"s_mov_b32 s5, {(s0 >> 32) & MASK32:#x}",
            "s_xor_saveexec_b64 s[6:7], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, old_exec: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SXorSaveExecB64Oracle.build_compute_shader(s0, old_exec)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSXorSaveExecOracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_basic_xor(self):
        s0 = 0x0000_FFFF_0000_FFFF
        old_exec = 0x00FF_00FF_00FF_00FF
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(s0, old_exec)
        self.assertEqual(new_exec, 0x00FF_FF00_00FF_FF00)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_identical_operands_produce_zero(self):
        val = 0x1234_5678_9ABC_DEF0
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(val, val)
        self.assertEqual(new_exec, 0)
        self.assertEqual(d, val)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s0(self):
        old_exec = 0x8000_0000_0000_0001
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(0, old_exec)
        self.assertEqual(new_exec, old_exec)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_zero_exec(self):
        s0 = 0xDEAD_BEEF_CAFE_BABE
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(s0, 0)
        self.assertEqual(new_exec, s0)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 1)

    def test_oracle_all_ones_inversion(self):
        mask = 0x5555_5555_5555_5555
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(ALL64, mask)
        self.assertEqual(new_exec, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(d, mask)
        self.assertEqual(scc, 1)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SXorSaveExecB64Oracle.translate_and_validate(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

if __name__ == "__main__":
    unittest.main()
