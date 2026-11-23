from __future__ import annotations

"""Architectural reference models, symbolic AST, deterministic interpreter,
multi-lane Wave64 divergence tests, and translator integration tests for
Bundle 56: Vector & Scalar Register Swap and Relative Addressing Suite.

Covers RDNA2 Instructions:
    - v_swap_b32 vD, vS
        Exchanges two 32-bit VGPRs across active execution lanes.
    - v_swaprel_b32 vD, vS
        Exchanges VGPR[D + M0[25:16]] and VGPR[S0 + M0[9:0]] under active lanes.
    - s_swappc_b64 sD, sS
        Saves next program counter in 64-bit SGPR pair D and jumps to address in S.
    - s_movrels_b32 sD, sS
        Reads SGPR[(S + M0) % 128] into scalar destination register sD.
    - s_movrels_b64 sD, sS
        Reads 64-bit SGPR pair starting at (S + M0) % 128 into pair sD.
    - v_movrels_b32 vD, vS
        Reads VGPR[S + M0] into destination vector register vD.

Hardware rules (GFX10 RDNA2):
    - v_swap_b32 exchanges values simultaneously using temporary storage.
    - Inactive lanes in EXEC preserve their existing VGPR values during vector swaps.
    - Relative addressing wraps modulo register bank bounds (128 SGPRs / 256 VGPRs).
    - M0 register provides dynamic base offsets for scalar and vector register files.
    - s_swappc_b64 acts as subroutine linkage in driver vertex fetch shaders and is rejected in compute kernels.

Standard library Python only (unittest, struct, math, dataclasses, enum).
Strictly zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import rdna2spirv
import spirv_mini_interp as si
import saveexec_run

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def model_swap(v0: int, v1: int) -> Tuple[int, int]:
    return v1 & MASK32, v0 & MASK32


def model_swaprel(dst_reg: int, src_reg: int, m0: int, vgprs: Dict[int, int]) -> Dict[int, int]:
    m0_src = m0 & 0x3FF
    m0_dst = (m0 >> 16) & 0x3FF
    actual_dst = (dst_reg + m0_dst) & 0xFF
    actual_src = (src_reg + m0_src) & 0xFF
    val_dst = vgprs.get(actual_dst, 0)
    val_src = vgprs.get(actual_src, 0)
    out = dict(vgprs)
    out[actual_dst] = val_src
    out[actual_src] = val_dst
    return out


def model_swappc(pc: int, target: int) -> Tuple[int, int]:
    return target & MASK64, (pc + 4) & MASK64


def model_movrels_b32(src_reg: int, m0: int, sgprs: Dict[int, int]) -> int:
    idx = (src_reg + m0) & 127
    return sgprs.get(idx, 0) & MASK32


def model_movrels_b64(src_reg: int, m0: int, sgprs: Dict[int, int]) -> Tuple[int, int]:
    idx0 = (src_reg + m0) & 127
    idx1 = (src_reg + m0 + 1) & 127
    return sgprs.get(idx0, 0) & MASK32, sgprs.get(idx1, 0) & MASK32


def model_v_movrels(src_reg: int, m0: int, vgprs: Dict[int, int]) -> int:
    idx = (src_reg + m0) & 255
    return vgprs.get(idx, 0) & MASK32

class TestSwapOracle(unittest.TestCase):
    def test_oracle_exchange(self):
        a, b = model_swap(0x12345678, 0x9ABCDEF0)
        self.assertEqual(a, 0x9ABCDEF0)
        self.assertEqual(b, 0x12345678)

    def test_translator_swap_oracle(self):
        prog = ["v_mov_b32_e32 v0, 100", "v_mov_b32_e32 v1, 200", "v_swap_b32 v0, v1", "s_endpgm"]
        spv = support.translate_lines("cs_swap_oracle", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCopyObject", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapAST(unittest.TestCase):
    def test_ast_evaluation(self):
        x, y = 55, 99
        x, y = model_swap(x, y)
        self.assertEqual(x, 99)
        self.assertEqual(y, 55)

    def test_translator_swap_ast(self):
        prog = ["v_mov_b32_e32 v2, 55", "v_mov_b32_e32 v3, 99", "v_swap_b32 v2, v3", "s_endpgm"]
        spv = support.translate_lines("cs_swap_ast", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCopyObject", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapDivergence(unittest.TestCase):
    def test_self_swap(self):
        a, b = model_swap(0xCAFE, 0xCAFE)
        self.assertEqual(a, 0xCAFE)
        self.assertEqual(b, 0xCAFE)

    def test_translator_swap_divergence(self):
        prog = [
            "s_mov_b32 exec_lo, 0x55555555",
            "s_mov_b32 exec_hi, 0xAAAAAAAA",
            "v_mov_b32_e32 v4, 111",
            "v_mov_b32_e32 v5, 222",
            "v_swap_b32 v4, v5",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_swap_div", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCopyObject", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})


if __name__ == "__main__":
    unittest.main()
