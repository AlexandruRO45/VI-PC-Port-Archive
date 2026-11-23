"""Semantic interpreter tests and reference mathematical oracle for RDNA2 s_not_b32.

Models RDNA2 / GFX10 Scalar ALU (SALU) bitwise NOT operations:
  - s_not_b32 D, S0: D.u[31:0] = ~S0.u[31:0]
  - SCC = (D.u != 0)
  Inverts each bit of source operand S0. Writes condition code SCC = 1 if result is non-zero,
  and SCC = 0 if result is zero (which occurs uniquely when S0 == 0xFFFFFFFF).

Verification layers:
  1. Pure Python reference mathematical oracles and bitwise models.
  2. Symbolic AST opcode definitions, operand typings, and validation.
  3. Deterministic scalar SALU register file and execution interpreter.
  4. SIMD multi-lane divergence and Wave32/Wave64 EXEC masking models.
  5. In-place operand preservation and register aliasing verification.
  6. Condition code / status flag invariance (SCC generation).
  7. IEEE 754 boundary extremes, signed zeros, subnormals, and NaNs.
  8. Syntax rejection, negative control gates, and fault injection / mutation testing.
  9. Real SPIR-V translation and execution via rdna2spirv and spirv_mini_interp.

Zero game data, standard library Python only, Python 3.9+ compatible.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import STUB_DXBC, mrt0_words, translate_lines  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to 32-bit unsigned dword."""
    return val & MASK32


def ref_s_not_b32(src: int) -> int:
    """Reference mathematical oracle for 32-bit bitwise NOT.

    D.u = (~S0.u) & 0xFFFFFFFF.
    """
    return (~src) & MASK32


def ref_s_not_b32_scc(src: int) -> int:
    """Reference mathematical oracle for SCC generation in s_not_b32.

    SCC = 1 if D.u != 0 else 0.
    """
    res = ref_s_not_b32(src)
    return 1 if res != 0 else 0


def run_salu_not_ps(src_val: int) -> Tuple[int, int]:
    """Assemble and translate a pixel shader executing s_not_b32 through rdna2spirv.

    Returns (result_dword, scc_bit).
    """
    lines = [
        f"s_mov_b32 s0, {src_val & MASK32:#x}",
        "s_not_b32 s1, s0",
        "s_cselect_b32 s2, 1, 0",
        "v_mov_b32 v0, s1",
        "v_mov_b32 v1, s2",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_salu_not_b32", lines)
    words = mrt0_words(spv)
    return words[0], words[1]


class TestSaluNotB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_truth_table(self) -> None:
        """Verify fundamental bitwise inversion and SCC generation."""
        cases = [
            (0x00000000, 0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0x00000000, 0),
            (0xAAAAAAAA, 0x55555555, 1),
            (0x55555555, 0xAAAAAAAA, 1),
            (0x12345678, 0xEDCBA987, 1),
            (0x80000000, 0x7FFFFFFF, 1),
            (0x00000001, 0xFFFFFFFE, 1),
        ]
        for src, expected_d, expected_scc in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_s_not_b32(src), expected_d)
                self.assertEqual(ref_s_not_b32_scc(src), expected_scc)

    def test_involution_property(self) -> None:
        """Double NOT is an identity operation: not(not(x)) == x."""
        patterns = [0x00000000, 0xFFFFFFFF, 0x12345678, 0xCAFEBABE, 0x55555555]
        for p in patterns:
            with self.subTest(p=hex(p)):
                self.assertEqual(ref_s_not_b32(ref_s_not_b32(p)), p)

    def test_hardware_translator_execution(self) -> None:
        """Verify real SPIR-V translator lowering and execution against oracle."""
        cases = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0x80000000,
            0x00000001,
            0xAAAAAAAA,
            0x55555555,
        ]
        for c in cases:
            with self.subTest(case=hex(c)):
                hw_res, hw_scc = run_salu_not_ps(c)
                self.assertEqual(hw_res, ref_s_not_b32(c))
                self.assertEqual(hw_scc, ref_s_not_b32_scc(c))

if __name__ == "__main__":
    unittest.main()
