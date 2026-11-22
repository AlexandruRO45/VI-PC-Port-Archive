from __future__ import annotations

import enum
import math
import struct
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# AMD RDNA2 / GFX10 GWS (Global Wavefront Sync) Architectural Constants
GWS_NUM_RESOURCES: int = 64  # Hardware semaphore / barrier resource slots (0..63)
GWS_RESOURCE_ID_MASK: int = 0x3F  # 6-bit resource index mask
SEMAPHORE_MAX_VALUE: int = 0xFFFF_FFFF  # 32-bit unsigned wrapping counter
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


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

    def bulk_release_br(self, res_id: int, release_count: int = 1) -> Tuple[int, List[int]]:
        """Executes a semaphore bulk release (broadcast/bulk signal) operation.

        Args:
            res_id: GWS resource index [0..63].
            release_count: Number of permits delivered in the bulk release.

        Returns:
            Tuple of (new_counter_value, list_of_awakened_wavefront_ids).
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_br")
        res.res_type = GWSResourceType.SEMAPHORE

        count = release_count & MASK32
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

        return res.value, awakened

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
    return (base + offset) & GWS_RESOURCE_ID_MASK


def ref_ds_gws_sema_br(subsystem: GWSSubsystem, m0_val: int, offset: int, release_count: int = 1) -> Tuple[int, List[int]]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_br."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.bulk_release_br(rid, release_count=release_count)


class TestGWSSemaBrOracleAndModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_br."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_bulk_release_increments_counter(self) -> None:
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=0, offset=0, release_count=1)
        self.assertEqual(val, 1)
        self.assertEqual(awakened, [])
        self.assertEqual(self.gws.resources[0].value, 1)

    def test_multi_count_bulk_release(self) -> None:
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=2, offset=3, release_count=16)
        self.assertEqual(val, 16)
        self.assertEqual(awakened, [])
        self.assertEqual(self.gws.resources[5].value, 16)
        self.assertEqual(self.gws.resources[5].signal_count, 16)

    def test_bulk_release_partial_waiters(self) -> None:
        rid = 8
        for w in [10, 20, 30, 40, 50]:
            self.gws.wait_p(rid, wave_id=w)

        # Bulk release count 3 releases first 3 waiters in FIFO order
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=3)
        self.assertEqual(awakened, [10, 20, 30])
        self.assertEqual(val, 0)
        self.assertEqual(self.gws.resources[rid].wait_queue, [40, 50])

    def test_bulk_release_exact_waiters(self) -> None:
        rid = 12
        for w in [1, 2, 3]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=3)
        self.assertEqual(awakened, [1, 2, 3])
        self.assertEqual(val, 0)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_bulk_release_exceeding_waiters_accumulates_counter(self) -> None:
        rid = 15
        for w in [101, 102]:
            self.gws.wait_p(rid, wave_id=w)

        # Release count 5 wakes 2 waiters, remaining 3 permits increment counter
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=5)
        self.assertEqual(awakened, [101, 102])
        self.assertEqual(val, 3)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_counter_wrapping_modulo_32bit(self) -> None:
        rid = 20
        self.gws.init_resource(rid, init_val=0xFFFF_FFF0)
        val, _ = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=20)
        # 0xFFFFFFF0 + 20 wraps to 4 modulo 2^32
        self.assertEqual(val, 4)
        self.assertEqual(self.gws.resources[rid].value, 4)

    def test_resource_id_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=62, offset=2), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=8), 4)

    def test_resource_isolation(self) -> None:
        ref_ds_gws_sema_br(self.gws, m0_val=1, offset=0, release_count=5)
        ref_ds_gws_sema_br(self.gws, m0_val=2, offset=0, release_count=10)

        self.assertEqual(self.gws.resources[1].value, 5)
        self.assertEqual(self.gws.resources[2].value, 10)
        self.assertEqual(self.gws.resources[0].value, 0)


if __name__ == "__main__":
    unittest.main()
