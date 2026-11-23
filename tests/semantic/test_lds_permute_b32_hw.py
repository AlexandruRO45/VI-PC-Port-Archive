"""Semantic interpreter tests and reference mathematical oracle for ds_permute_b32.

RDNA2 / GFX10 LDS cross-lane forward (scatter) permute:
    ds_permute_b32 vdst, vaddr, vdata [offset:offset16]

Semantics:
    Every source lane i active in EXEC calculates:
        effective_address = (vaddr + offset16) & 0xFFFFFFFF
        target_lane = ((effective_address >> 2) & 31) (within its 32-lane half)
    and sends its vdata to target_lane.
    - Destination lanes that receive no data become 0 (when active in EXEC).
    - Destination lanes that are inactive in EXEC retain their old vdst.
    - In wave64, operations occur independently within half 0 (lanes 0..31) and half 1 (lanes 32..63).
    - Collisions: If multiple active lanes target the same destination, the RDNA2 ISA states
      the result is one of the writers. Leonida translator resolves this with highest-numbered writer winning.
"""
from __future__ import annotations

import math
import os
import re
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
OFFSET16_MASK: int = 0xFFFF
OLD_VDST: int = 0x5A5A1234
ALL32: int = 0xFFFFFFFF
ALL64: int = (1 << 64) - 1


def calc_target_lane(vaddr: int, offset: int = 0) -> int:
    """Calculate target lane index within a 32-lane half."""
    eff = (vaddr + (offset & OFFSET16_MASK)) & MASK32
    return (eff >> 2) & 31


def ref_ds_permute_scalar(vaddr: int, vdata: int, offset: int = 0) -> tuple[int, int]:
    """Pure mathematical single-lane target and masked 32-bit data."""
    return calc_target_lane(vaddr, offset), vdata & MASK32


def ref_ds_permute_wave(
    addrs: list[int],
    datas: list[int],
    exec_mask: int = ALL32,
    offset: int = 0,
    lanes: int = 32,
    old_dsts: list[int] | None = None,
) -> list[int]:
    """Pure reference simulation of ds_permute_b32 across wave32 or wave64.

    Matches Leonida translator semantics where highest active source lane wins on collision.
    """
    if old_dsts is None:
        old_dsts = [OLD_VDST] * lanes
    out = list(old_dsts)

    for half_base in range(0, lanes, 32):
        half_len = min(32, lanes - half_base)
        writers: dict[int, list[int]] = {d: [] for d in range(32)}

        for i in range(half_len):
            src_lane = half_base + i
            if (exec_mask >> src_lane) & 1:
                target = calc_target_lane(addrs[src_lane], offset)
                writers[target].append(datas[src_lane] & MASK32)

        for i in range(half_len):
            dst_lane = half_base + i
            if (exec_mask >> dst_lane) & 1:
                if writers[i]:
                    out[dst_lane] = writers[i][-1]
                else:
                    out[dst_lane] = 0
            else:
                out[dst_lane] = old_dsts[dst_lane]

    return out


class ArchitecturalPermuteModel:
    """Architectural model verifying forward cross-lane permute and collision sets."""

    def __init__(self, lanes: int = 32) -> None:
        self.lanes = lanes

    def execute(
        self,
        addrs: list[int],
        datas: list[int],
        exec_mask: int,
        offset: int = 0,
        old_dsts: list[int] | None = None,
    ) -> list[int]:
        return ref_ds_permute_wave(addrs, datas, exec_mask, offset, self.lanes, old_dsts)

    def allowed_outcomes(
        self,
        addrs: list[int],
        datas: list[int],
        exec_mask: int,
        offset: int = 0,
        old_dsts: list[int] | None = None,
    ) -> list[set[int]]:
        """Returns the set of all valid values per lane permitted by the RDNA2 ISA."""
        if old_dsts is None:
            old_dsts = [OLD_VDST] * self.lanes
        out: list[set[int]] = []

        for half_base in range(0, self.lanes, 32):
            half_len = min(32, self.lanes - half_base)
            writers: dict[int, list[int]] = {d: [] for d in range(32)}

            for i in range(half_len):
                src_lane = half_base + i
                if (exec_mask >> src_lane) & 1:
                    target = calc_target_lane(addrs[src_lane], offset)
                    writers[target].append(datas[src_lane] & MASK32)

            for i in range(half_len):
                dst_lane = half_base + i
                if (exec_mask >> dst_lane) & 1:
                    if writers[i]:
                        out.append(set(writers[i]))
                    else:
                        out.append({0})
                else:
                    out.append({old_dsts[dst_lane]})

        return out


class TestDsPermuteOracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Mathematical reference oracle and architectural model tests."""

    def test_target_lane_calculation_basic(self) -> None:
        # Address in bytes: dword 0 is byte 0, dword 1 is byte 4, etc.
        self.assertEqual(calc_target_lane(0), 0)
        self.assertEqual(calc_target_lane(4), 1)
        self.assertEqual(calc_target_lane(8), 2)
        self.assertEqual(calc_target_lane(124), 31)
        # Modulo 32 wrapping: dword 32 is byte 128 -> lane 0
        self.assertEqual(calc_target_lane(128), 0)
        self.assertEqual(calc_target_lane(132), 1)

    def test_target_lane_offset_addition(self) -> None:
        # Address 0 with offset 4 -> lane 1
        self.assertEqual(calc_target_lane(0, offset=4), 1)
        # Address 4 with offset 8 -> byte 12 -> lane 3
        self.assertEqual(calc_target_lane(4, offset=8), 3)
        # Unaligned low 2 bits are ignored by >> 2
        self.assertEqual(calc_target_lane(1, offset=0), 0)
        self.assertEqual(calc_target_lane(2, offset=0), 0)
        self.assertEqual(calc_target_lane(3, offset=0), 0)
        self.assertEqual(calc_target_lane(5, offset=0), 1)

    def test_reference_oracle_unwritten_lanes_zero(self) -> None:
        # Only lane 0 writes to lane 5; all other lanes remain unwritten
        addrs = [20 if i == 0 else 0 for i in range(32)]
        datas = [0xABCD if i == 0 else 0 for i in range(32)]
        # EXEC has all 32 lanes on, but only lane 0 is an active writer (exec_mask=0x1)
        res = ref_ds_permute_wave(addrs, datas, exec_mask=0x1, offset=0, lanes=32)
        # Lane 5 receives 0xABCD, unwritten lanes (0..4, 6..31) are off in EXEC so they keep OLD_VDST
        self.assertEqual(res[5], OLD_VDST)  # since lane 5 is off in EXEC (exec_mask=1 only has bit 0 on)
        self.assertEqual(res[0], 0)         # lane 0 is active, nobody wrote to it, gets 0

        # Now all lanes on in EXEC:
        res_all = ref_ds_permute_wave(addrs, datas, exec_mask=ALL32, offset=0, lanes=32)
        self.assertEqual(res_all[5], 0xABCD)
        for d in range(32):
            if d != 5 and d != 0:
                self.assertEqual(res_all[d], 0)

    def test_reference_oracle_collision_highest_writer_wins(self) -> None:
        # All 32 lanes write to lane 0 with data = 100 + lane
        addrs = [0] * 32
        datas = [100 + i for i in range(32)]
        res = ref_ds_permute_wave(addrs, datas, exec_mask=ALL32, offset=0, lanes=32)
        self.assertEqual(res[0], 100 + 31)  # lane 31 is highest writer
        for d in range(1, 32):
            self.assertEqual(res[d], 0)

    def test_architectural_model_wave64_halves_isolation(self) -> None:
        model = ArchitecturalPermuteModel(lanes=64)
        # Lower half lanes write to lane 10; upper half lanes write to lane 10 of upper half (lane 42)
        addrs = [40] * 64  # byte 40 -> lane 10
        datas = [1000 + i for i in range(64)]
        res = model.execute(addrs, datas, exec_mask=ALL64, offset=0)
        self.assertEqual(res[10], 1000 + 31)
        self.assertEqual(res[42], 1000 + 63)
        self.assertEqual(res[0], 0)
        self.assertEqual(res[32], 0)

if __name__ == "__main__":
    unittest.main()
