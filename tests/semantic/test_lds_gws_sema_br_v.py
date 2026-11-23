from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 ds_gws_sema_br_v.

Models AMD RDNA2 / GFX10 Global Wavefront Sync (GWS) hardware semaphore
branch-release operation (bulk signal with branch / wakeup notification):
  - ds_gws_sema_br_v: Releases permits to the targeted GWS semaphore resource.
    Awakens waiting wavefronts from the FIFO wait queue up to the released
    permit count. Any unconsumed permits increment the resource counter.
    Signals branch taken / SCC=1 if one or more wavefronts were awakened.
    Signals branch not taken / SCC=0 if no wavefronts were waiting.

RDNA2 GWS Architectural Rules:
  1. GWS Resource Addressing:
     - 64 hardware resource slots (0..63) in GDS space.
     - Effective target resource ID: (m0 + (offset16 & 0xFFFF)) & 0x3F.
  2. Semaphore Branch Release Semantics:
     - Release Count: Specified via vdata (VGPR) or defaults to 1 permit.
     - Waiter Awakening: Waiting wavefronts are dequeued in strict FIFO order.
     - Counter Accumulation: Unconsumed permits add to the 32-bit unsigned counter.
     - Branch Condition: Evaluates to True (SCC=1) if waiters were awakened.
  3. Resource Isolation & Mode Enforcement:
     - Each resource slot is independently configured (UNINITIALIZED, SEMAPHORE, BARRIER).
     - Executing ds_gws_sema_br_v on a BARRIER resource triggers an architectural type fault.

Standard library Python only (python -I, struct, math, unittest). Zero game data.
"""

import enum
import math
import struct
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# RDNA2 GWS Architectural Constants
GWS_NUM_RESOURCES: int = 64
GWS_RESOURCE_ID_MASK: int = 0x3F
OFFSET16_MASK: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


class GWSResourceType(enum.Enum):
    """GWS hardware resource mode."""
    UNINITIALIZED = "uninitialized"
    SEMAPHORE = "semaphore"
    BARRIER = "barrier"


@dataclass
class GWSSemaphoreResource:
    """Models a single hardware GWS semaphore resource in RDNA2."""
    res_id: int
    res_type: GWSResourceType = GWSResourceType.SEMAPHORE
    value: int = 0
    wait_queue: List[int] = field(default_factory=list)
    init_count: int = 0
    signal_count: int = 0
    wake_count: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.res_id < GWS_NUM_RESOURCES):
            raise ValueError(f"GWS resource id out of bounds [0, 63]: {self.res_id}")
        self.value &= MASK32


class GWSSubsystem:
    """Models the complete Global Wavefront Sync hardware block with 64 slots."""

    def __init__(self) -> None:
        self.resources: List[GWSSemaphoreResource] = [
            GWSSemaphoreResource(res_id=i) for i in range(GWS_NUM_RESOURCES)
        ]

    def reset(self) -> None:
        self.resources = [
            GWSSemaphoreResource(res_id=i) for i in range(GWS_NUM_RESOURCES)
        ]

    def init_resource(self, res_id: int, init_val: int = 0, res_type: GWSResourceType = GWSResourceType.SEMAPHORE) -> None:
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        res.res_type = res_type
        res.value = init_val & MASK32
        res.init_count = init_val & MASK32
        res.wait_queue.clear()
        res.signal_count = 0
        res.wake_count = 0

    def bulk_release_br_v(self, res_id: int, release_count: int = 1) -> Tuple[int, List[int], bool]:
        """Executes a semaphore branch release (bulk signal with branch indicator).

        Args:
            res_id: GWS resource index [0..63].
            release_count: Number of permits delivered in the bulk release.

        Returns:
            Tuple of (new_counter_value, list_of_awakened_wavefront_ids, branch_taken).
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_br_v")
        res.res_type = GWSResourceType.SEMAPHORE

        count = max(1, release_count & MASK32)
        res.signal_count += count
        awakened: List[int] = []

        if res.wait_queue:
            num_to_wake = min(count, len(res.wait_queue))
            awakened = res.wait_queue[:num_to_wake]
            res.wait_queue = res.wait_queue[num_to_wake:]
            res.wake_count += len(awakened)
            remaining_permits = count - num_to_wake
            res.value = (res.value + remaining_permits) & MASK32
        else:
            res.value = (res.value + count) & MASK32

        branch_taken = (len(awakened) > 0)
        return res.value, awakened, branch_taken

    def wait_p(self, res_id: int, wave_id: int) -> bool:
        """Executes a semaphore P (wait/decrement/acquire) operation."""
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_p")
        res.res_type = GWSResourceType.SEMAPHORE

        if res.value > 0:
            res.value = (res.value - 1) & MASK32
            return True
        else:
            res.wait_queue.append(wave_id)
            return False


def ref_gws_resource_id(m0_val: int, offset: int) -> int:
    """Calculates target GWS resource index from M0 base register and offset."""
    base = m0_val & GWS_RESOURCE_ID_MASK
    return (base + (offset & OFFSET16_MASK)) & GWS_RESOURCE_ID_MASK


def ref_ds_gws_sema_br_v(subsystem: GWSSubsystem, m0_val: int, offset: int, release_count: int = 1) -> Tuple[int, List[int], bool]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_br_v."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.bulk_release_br_v(rid, release_count=release_count)


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_br_v."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_release_increments_counter(self) -> None:
        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=0, offset=0, release_count=1)
        self.assertEqual(val, 1)
        self.assertEqual(awakened, [])
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[0].value, 1)

    def test_multi_permit_release_accumulates(self) -> None:
        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=2, offset=3, release_count=16)
        self.assertEqual(val, 16)
        self.assertEqual(awakened, [])
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[5].value, 16)

    def test_release_wakes_waiting_wavefronts_fifo(self) -> None:
        rid = 8
        for w in [10, 20, 30]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=3)
        self.assertEqual(awakened, [10, 20, 30])
        self.assertEqual(val, 0)
        self.assertTrue(branch)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_release_partial_waiters(self) -> None:
        rid = 10
        for w in [1, 2, 3, 4, 5]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=2)
        self.assertEqual(awakened, [1, 2])
        self.assertEqual(val, 0)
        self.assertTrue(branch)
        self.assertEqual(self.gws.resources[rid].wait_queue, [3, 4, 5])

    def test_release_exceeding_waiters_accumulates_remainder(self) -> None:
        rid = 12
        for w in [101, 102]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=5)
        self.assertEqual(awakened, [101, 102])
        self.assertEqual(val, 3)
        self.assertTrue(branch)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_counter_wrapping_modulo_32bit(self) -> None:
        rid = 15
        self.gws.init_resource(rid, init_val=0xFFFF_FFF0)
        val, _, _ = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=20)
        self.assertEqual(val, 4)
        self.assertEqual(self.gws.resources[rid].value, 4)

    def test_resource_id_modulo_64_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)
        self.assertEqual(ref_gws_resource_id(m0_val=0, offset=0x10005), 5)

    def test_resource_isolation(self) -> None:
        ref_ds_gws_sema_br_v(self.gws, m0_val=1, offset=0, release_count=5)
        ref_ds_gws_sema_br_v(self.gws, m0_val=2, offset=0, release_count=10)

        self.assertEqual(self.gws.resources[1].value, 5)
        self.assertEqual(self.gws.resources[2].value, 10)
        self.assertEqual(self.gws.resources[0].value, 0)

    def test_barrier_mode_rejection(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=0, res_type=GWSResourceType.BARRIER)
        with self.assertRaises(TypeError):
            ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=1)


if __name__ == "__main__":
    unittest.main()
