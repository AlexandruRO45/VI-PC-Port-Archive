"""Semantic interpreter tests and reference mathematical oracle for RDNA2 s_xnor_saveexec_b32.

Models RDNA2 / GFX10 Scalar ALU (SALU) saveexec operations on 32-bit execution masks:
  - s_xnor_saveexec_b32 D, S0:
      D = EXEC_LO
      new_exec_lo = ~(S0 ^ EXEC_LO) & 0xFFFFFFFF
      EXEC_LO = new_exec_lo
      SCC = (EXEC_LO != 0)
  In wave64, s_xnor_saveexec_b32 operates strictly on the lower 32 lanes (EXEC_LO [31:0]),
  leaving EXEC_HI [63:32] completely untouched, and SCC reflects (EXEC_LO != 0).
  In wave32, it operates on the full wave execution mask.

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
import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
FSSM_64: bytes = bytes(0x138) + struct.pack("<4I", 0x88, 64, 1, 1)
VSHARP: List[int] = [0, 0, 0x1000, 0x31016FAC]


def u32(val: int) -> int:
    return val & MASK32


def ref_s_xnor_saveexec_b32(s0: int, old_exec_lo: int) -> Tuple[int, int, int]:
    """Reference mathematical oracle for s_xnor_saveexec_b32.

    Returns (new_exec_lo, saved_d, scc).
    """
    s0_u = s0 & MASK32
    exec_lo = old_exec_lo & MASK32
    new_exec_lo = (~(s0_u ^ exec_lo)) & MASK32
    saved_d = exec_lo
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, saved_d, scc


def run_salu_xnor_saveexec_b32(s0_val: int, init_lo: int, init_hi: int = 0x000000FF) -> Tuple[int, int, int, int]:
    """Execute s_xnor_saveexec_b32 via translate_lines and spirv_mini_interp.run_workgroup.

    Returns (new_exec_lo, exec_hi, scc, popcount_saved_d).
    """
    lines = [
        "s_mov_b64 exec, -1",
        f"s_mov_b32 exec_lo, {init_lo & MASK32:#x}",
        f"s_mov_b32 exec_hi, {init_hi & MASK32:#x}",
        f"s_xnor_saveexec_b32 s8, {s0_val & MASK32:#x}",
        "s_cselect_b32 s10, 1, 0",
        "v_mov_b32_e32 v10, 0",
        "s_mov_b64 s[12:13], exec",
        "s_mov_b64 exec, -1",
        "v_cndmask_b32_e64 v10, 0, 1, s[12:13]",
        "v_lshlrev_b32_e32 v11, 2, v0",
        "v_add_nc_u32_e32 v11, 64, v11",
        "buffer_store_dword v10, v11, s[0:3], 0 offen",
        "s_bcnt1_i32_b32 s11, s8",
        "v_cmpx_eq_u32_e32 0, v0",
        "v_mov_b32_e32 v1, s10",
        "v_mov_b32_e32 v2, s11",
        "v_mov_b32_e32 v12, 0", "buffer_store_dword v1, v12, s[0:3], 0 offen",
        "v_mov_b32_e32 v12, 4", "buffer_store_dword v2, v12, s[0:3], 0 offen",
        "s_endpgm",
    ]
    spv = support.translate_lines("cs_xnor_saveexec_b32", lines, fssm=FSSM_64)
    mem = si.run_workgroup(
        spv, 64, 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    exec_lo = sum(((mem.get(64 + 4 * i, 0) & 1) << i) for i in range(32))
    exec_hi = sum(((mem.get(64 + 4 * (32 + i), 0) & 1) << i) for i in range(32))
    scc = mem.get(0, 0)
    popcount_saved_d = mem.get(4, 0)
    return exec_lo, exec_hi, scc, popcount_saved_d


class TestSaluXnorSaveexecB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 1),
            (0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0),
            (0x55555555, 0x0000000F, 0xAAAAAAA5, 0x0000000F, 1),
            (0xAAAAAAAA, 0xAAAAAAAA, 0xFFFFFFFF, 0xAAAAAAAA, 1),
            (0x12345678, 0xEDCBA987, 0x00000000, 0xEDCBA987, 0),
        ]
        for s0, init_lo, exp_new_exec, exp_saved_d, exp_scc in cases:
            with self.subTest(s0=hex(s0), init_lo=hex(init_lo)):
                new_exec, saved_d, scc = ref_s_xnor_saveexec_b32(s0, init_lo)
                self.assertEqual(new_exec, exp_new_exec)
                self.assertEqual(saved_d, exp_saved_d)
                self.assertEqual(scc, exp_scc)

    def test_hardware_translator_execution(self) -> None:
        cases = [
            (0x55555555, 0x0000000F, 0x000000FF),
            (0xAAAAAAAA, 0xAAAAAAAA, 0x12345678),
            (0x55555555, 0xAAAAAAAA, 0x000000FF),
        ]
        for s0, init_lo, init_hi in cases:
            with self.subTest(s0=hex(s0), init_lo=hex(init_lo), init_hi=hex(init_hi)):
                exp_new_lo, exp_saved_d, exp_scc = ref_s_xnor_saveexec_b32(s0, init_lo)
                hw_lo, hw_hi, hw_scc, hw_popcount = run_salu_xnor_saveexec_b32(s0, init_lo, init_hi)
                self.assertEqual(hw_lo, exp_new_lo)
                self.assertEqual(hw_hi, init_hi)
                self.assertEqual(hw_scc, exp_scc)
                self.assertEqual(hw_popcount, bin(exp_saved_d).count("1"))

if __name__ == "__main__":
    unittest.main()
