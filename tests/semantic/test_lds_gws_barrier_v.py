from __future__ import annotations

"""Reference oracle and architectural model for ds_gws_barrier_v.

RDNA2 / GFX10 Data Share Global Wave Sync Barrier Release (ds_gws_barrier_v):
Signals and releases waiting wavefronts queued at a Global Wave Sync (GWS)
hardware barrier slot located in Global Data Share (GDS).

Syntax:
    ds_gws_barrier_v [vdata0] [offset:offset16] gds

Semantics:
    resource_id = (m0 + (offset16 & 0xFFFF)) & 0x3F
    first_active_lane = ctz(EXEC)
    if EXEC != 0:
        GWS[resource_id].releases += 1
        GWS[resource_id].arrived = 0
        woken_waves = GWS[resource_id].waiting_waves[:]
        GWS[resource_id].waiting_waves.clear()
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 GWS Architectural Constants
GWS_NUM_RESOURCES: int = 64         # 64 hardware GWS barrier slots on GFX10
GWS_RESOURCE_ID_MASK: int = 0x3F    # 6-bit resource ID (0..63)
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte/resource offset mask
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned dword mask
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def calc_gws_resource_id(offset: int = 0, m0: int = 0) -> int:
    """Calculate the 6-bit GWS resource identifier."""
    return (m0 + (offset & OFFSET16_MASK)) & GWS_RESOURCE_ID_MASK


def calc_gws_effective_offset(offset: int = 0, m0: int = 0) -> int:
    """Calculate effective offset before modulo masking."""
    return (m0 + (offset & OFFSET16_MASK)) & MASK32


def ref_ds_gws_barrier_v(arrived_count: int, releases: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_gws_barrier_v (signal/release)."""
    next_arrived = 0
    next_releases = u32(releases + 1)
    return next_arrived, next_releases


@dataclass
class GWSBarrierResource:
    """Architectural state representation of an individual GWS hardware barrier slot."""
    target_count: int = 1
    arrived_count: int = 0
    total_releases: int = 0
    waiting_waves: list[int] = field(default_factory=list)

    def arrive_and_wait(self, wave_id: int) -> None:
        self.arrived_count = u32(self.arrived_count + 1)
        self.waiting_waves.append(wave_id)

    def release_v(self) -> Tuple[int, list[int]]:
        self.total_releases = u32(self.total_releases + 1)
        self.arrived_count = 0
        woken = self.waiting_waves.copy()
        self.waiting_waves.clear()
        return self.total_releases, woken

    def reset(self) -> None:
        self.target_count = 1
        self.arrived_count = 0
        self.total_releases = 0
        self.waiting_waves.clear()


class GWSMemory:
    """Architectural memory model representing the Global Wave Sync (GWS) subsystem."""

    def __init__(self, num_resources: int = GWS_NUM_RESOURCES) -> None:
        if num_resources <= 0 or (num_resources & (num_resources - 1)) != 0:
            raise ValueError(f"num_resources must be a positive power of two, got {num_resources}")
        self.num_resources = num_resources
        self.mask = num_resources - 1
        self.barriers: list[GWSBarrierResource] = [
            GWSBarrierResource() for _ in range(num_resources)
        ]

    def read_arrived(self, resource_id: int) -> int:
        idx = resource_id & self.mask
        return self.barriers[idx].arrived_count

    def read_releases(self, resource_id: int) -> int:
        idx = resource_id & self.mask
        return self.barriers[idx].total_releases

    def barrier_v(
        self,
        offset: int = 0,
        m0: int = 0,
    ) -> Tuple[int, int, list[int]]:
        rid = calc_gws_resource_id(offset=offset, m0=m0)
        idx = rid & self.mask
        releases, woken = self.barriers[idx].release_v()
        return rid, releases, woken


class TestGWSBarrierVReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_barrier_v_release(self) -> None:
        next_arr, next_rel = ref_ds_gws_barrier_v(arrived_count=3, releases=0)
        self.assertEqual(next_arr, 0)
        self.assertEqual(next_rel, 1)

        next_arr2, next_rel2 = ref_ds_gws_barrier_v(arrived_count=0, releases=10)
        self.assertEqual(next_arr2, 0)
        self.assertEqual(next_rel2, 11)

    def test_oracle_32bit_releases_wrapping(self) -> None:
        next_arr, next_rel = ref_ds_gws_barrier_v(arrived_count=1, releases=0xFFFFFFFF)
        self.assertEqual(next_arr, 0)
        self.assertEqual(next_rel, 0)

    def test_resource_id_calculation(self) -> None:
        self.assertEqual(calc_gws_resource_id(0, 0), 0)
        self.assertEqual(calc_gws_resource_id(12, 0), 12)
        self.assertEqual(calc_gws_resource_id(63, 0), 63)
        self.assertEqual(calc_gws_resource_id(64, 0), 0)
        self.assertEqual(calc_gws_resource_id(65, 0), 1)
        self.assertEqual(calc_gws_resource_id(10, 5), 15)
        self.assertEqual(calc_gws_resource_id(60, 6), 2)

    def test_effective_offset_calculation(self) -> None:
        self.assertEqual(calc_gws_effective_offset(0x40, 0x10), 0x50)
        self.assertEqual(calc_gws_effective_offset(0xFFFF, 1), 0x10000)

    def test_architectural_model_release_and_unblock(self) -> None:
        gws = GWSMemory()
        gws.barriers[4].arrive_and_wait(101)
        gws.barriers[4].arrive_and_wait(102)
        gws.barriers[4].arrive_and_wait(103)
        self.assertEqual(gws.read_arrived(4), 3)
        self.assertEqual(gws.read_releases(4), 0)

        rid, rels, woken = gws.barrier_v(offset=4)
        self.assertEqual(rid, 4)
        self.assertEqual(rels, 1)
        self.assertEqual(woken, [101, 102, 103])
        self.assertEqual(gws.read_arrived(4), 0)
        self.assertEqual(gws.read_releases(4), 1)

    def test_barrier_reset(self) -> None:
        gws = GWSMemory()
        gws.barriers[2].arrive_and_wait(99)
        gws.barriers[2].release_v()
        gws.barriers[2].reset()
        self.assertEqual(gws.read_arrived(2), 0)
        self.assertEqual(gws.read_releases(2), 0)
        self.assertEqual(len(gws.barriers[2].waiting_waves), 0)


if __name__ == "__main__":
    unittest.main()
