from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 VALU Quad / Multi-lane Reduction & LDS Swizzle Extensions Suite.

Covers:
- s_quadmask_b32: 32-bit scalar nibble/quad reduction mask generation (8 bits).
- s_quadmask_b64: 64-bit scalar quad reduction mask generation (16 bits).
- v_perm_b32: 32-bit vector byte permutation from 64-bit pair with sign extension/constant injection.
- ds_swizzle_b32: LDS cross-lane subgroup swizzle (QUAD_PERM, SWAP, REVERSE, BROADCAST).
- ds_bpermute_b32: LDS cross-lane backward gather permute.
- ds_permute_b32: LDS cross-lane forward scatter permute.

Standard library only (python -I, struct, math, unittest). Zero game data.
All test suites execute real translated assembly via support.translate_lines and spirv_mini_interp.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFF_FFFF
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_shader_words(name: str, lines: List[str]) -> Tuple[int, str]:
    full_lines = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = support.translate_lines(name, full_lines)
    words = support.mrt0_words(spv)
    return words[0], spv


def run_workgroup_lanes(name: str, lines: List[str], lanes: int = 4) -> Tuple[List[int], str]:
    full_lines = lines + ["exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
    spv = support.translate_lines(name, full_lines)
    mem: Dict[int, int] = {}
    si.run_workgroup(
        spv, lanes, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory=mem
    )
    res = [mem.get(4 * i, 0) for i in range(lanes)]
    return res, spv


# ------------------------------------------------------------------------------
# s_quadmask_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_quadmask_b32(s0: int) -> int:
    s0 = s0 & MASK32
    res = 0
    for i in range(8):
        if (s0 >> (4 * i)) & 0xF:
            res |= (1 << i)
    return res


class TestQuadmaskB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            0x12,
            0xF0F0_F0F0,
            0x0000_0001,
            0x8000_0000,
            0xFFFF_FFFF,
            0x0000_0000,
        ]
        for idx, val in enumerate(vectors):
            val_hw, _ = run_shader_words(f"ps_qm32_oracle_{idx}", [
                f"s_mov_b32 s1, {hex(val)}",
                "s_quadmask_b32 s0, s1",
                "v_mov_b32 v0, s0",
            ])
            self.assertEqual(val_hw, ref_quadmask_b32(val))

    def test_oracle_quad_boundaries_hardware(self):
        # Test individual nibbles
        for i in range(8):
            val = 1 << (4 * i)
            val_hw, _ = run_shader_words(f"ps_qm32_nibble_{i}", [
                f"s_mov_b32 s1, {hex(val)}",
                "s_quadmask_b32 s0, s1",
                "v_mov_b32 v0, s0",
            ])
            self.assertEqual(val_hw, 1 << i)
