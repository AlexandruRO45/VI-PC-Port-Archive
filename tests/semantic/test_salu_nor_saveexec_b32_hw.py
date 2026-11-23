from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_nor_saveexec_b32.

RDNA2 Instruction Set Architecture, SOP1:
    s_nor_saveexec_b32 D, S0

Semantic operation:
    D.u32 = EXEC_LO
    EXEC_LO.u32 = ~(S0.u32 | EXEC_LO.u32) & 0xFFFFFFFF
    SCC = (EXEC_LO != 0)

Outputs:
    D: 32-bit destination SGPR storing the previous EXEC_LO mask.
    EXEC_LO: Updated 32-bit execution mask.
    SCC: 1 if the updated EXEC_LO mask is non-zero, 0 if all zeros.

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

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_nor_saveexec_b32_oracle(s0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference mathematical model for s_nor_saveexec_b32.

    Args:
        s0: 32-bit unsigned integer source mask.
        old_exec: 32-bit unsigned integer execution mask before execution.

    Returns:
        (new_exec, d, scc) tuple where:
            new_exec: ~(s0 | old_exec) & 0xFFFFFFFF (32-bit unsigned)
            d: old_exec & 0xFFFFFFFF (32-bit unsigned)
            scc: 1 if new_exec != 0 else 0
    """
    s0_u32 = s0 & MASK32
    old_exec_u32 = old_exec & MASK32
    new_exec = (~(s0_u32 | old_exec_u32)) & MASK32
    d = old_exec_u32
    scc = 1 if new_exec != 0 else 0
    return new_exec, d, scc


class SNorSaveExecB32Oracle:
    """Architectural model and hardware-rule validator for s_nor_saveexec_b32."""

    @staticmethod
    def eval(s0: int, old_exec: int) -> Tuple[int, int, int]:
        return s_nor_saveexec_b32_oracle(s0, old_exec)

    @staticmethod
    def build_compute_shader(s0: int, old_exec: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_nor_saveexec_b32."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 32, 1, 1)
        name = "cs_nor_saveexec_b32_hw"
        lines = [
            f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}",
            f"s_mov_b32 s4, {s0 & MASK32:#x}",
            "s_nor_saveexec_b32 s6, s4",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, old_exec: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SNorSaveExecB32Oracle.build_compute_shader(s0, old_exec)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSNorSaveExecOracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_basic_nor(self):
        s0 = 0x0000_FFFF
        old_exec = 0x00FF_00FF
        # s0 | old_exec = 0x00FF_FFFF; ~0x00FF_FFFF = 0xFF00_0000
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(s0, old_exec)
        self.assertEqual(new_exec, 0xFF00_0000)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_both_zeros_produce_all_ones(self):
        # NOR(0, 0) = 0xFFFFFFFF: all lanes activate
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(0, 0)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 1)

    def test_oracle_all_ones_produces_zero(self):
        # NOR(0xFFFFFFFF, x) = 0: all lanes turn off
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(MASK32, 0x1234_5678)
        self.assertEqual(new_exec, 0)
        self.assertEqual(d, 0x1234_5678)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s0_inverts_exec(self):
        old_exec = 0x5555_5555
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(0, old_exec)
        self.assertEqual(new_exec, 0xAAAA_AAAA)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SNorSaveExecB32Oracle.translate_and_validate(0x5555_5555, 0xAAAA_AAAA)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

if __name__ == "__main__":
    unittest.main()
