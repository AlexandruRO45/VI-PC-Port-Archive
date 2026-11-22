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

    def signal_v(self, res_id: int) -> Tuple[int, Optional[int]]:
        """Executes a semaphore V (signal/increment/post) operation.

        Returns:
            Tuple of (new_counter_value, awakened_wavefront_id_or_None).
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_v")
        res.res_type = GWSResourceType.SEMAPHORE

        res.signal_count += 1
        if res.wait_queue:
            awakened = res.wait_queue.pop(0)
            res.wake_count += 1
            return res.value, awakened
        else:
            res.value = (res.value + 1) & MASK32
            return res.value, None

    def wait_p(self, res_id: int, wave_id: int) -> bool:
        """Executes a semaphore P (wait/decrement/acquire) operation.

        Returns:
            True if acquired immediately, False if blocked and enqueued.
        """
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


def ref_ds_gws_sema_v(subsystem: GWSSubsystem, m0_val: int, offset: int) -> Tuple[int, Optional[int]]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_v."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.signal_v(rid)


class TestGWSSemaVOracleAndModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_v."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_signal_increments_counter(self) -> None:
        val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=0, offset=0)
        self.assertEqual(val, 1)
        self.assertIsNone(awakened)
        self.assertEqual(self.gws.resources[0].value, 1)
        self.assertEqual(self.gws.resources[0].signal_count, 1)

    def test_multiple_consecutive_signals(self) -> None:
        for expected in range(1, 101):
            val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=5, offset=2)
            self.assertEqual(val, expected)
            self.assertIsNone(awakened)
        self.assertEqual(self.gws.resources[7].value, 100)
        self.assertEqual(self.gws.resources[7].signal_count, 100)

    def test_counter_wrapping_modulo_32bit(self) -> None:
        rid = 12
        self.gws.init_resource(rid, init_val=0xFFFF_FFFE)
        val, _ = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertEqual(val, 0xFFFF_FFFF)

        val, _ = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertEqual(val, 0)
        self.assertEqual(self.gws.resources[rid].value, 0)

    def test_awaken_single_waiting_wavefront(self) -> None:
        rid = 3
        # Wavefront 42 attempts P on zero-counter semaphore -> blocks
        acquired = self.gws.wait_p(rid, wave_id=42)
        self.assertFalse(acquired)
        self.assertEqual(self.gws.resources[rid].wait_queue, [42])

        # ds_gws_sema_v signals: should wake wave 42, counter stays 0
        val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertEqual(awakened, 42)
        self.assertEqual(val, 0)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)
        self.assertEqual(self.gws.resources[rid].wake_count, 1)

    def test_awaken_fifo_queue_ordering(self) -> None:
        rid = 10
        for w in [101, 102, 103, 104]:
            self.gws.wait_p(rid, wave_id=w)

        # 4 signals should awaken in exact FIFO order
        for expected_wave in [101, 102, 103, 104]:
            val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
            self.assertEqual(awakened, expected_wave)
            self.assertEqual(val, 0)

        # 5th signal increments counter
        val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertIsNone(awakened)
        self.assertEqual(val, 1)

    def test_resource_id_wrapping(self) -> None:
        # Offset + M0 base wrapping across 64 slots
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)

    def test_independent_gws_resource_isolation(self) -> None:
        ref_ds_gws_sema_v(self.gws, m0_val=1, offset=0)
        ref_ds_gws_sema_v(self.gws, m0_val=2, offset=0)
        ref_ds_gws_sema_v(self.gws, m0_val=2, offset=0)

        self.assertEqual(self.gws.resources[1].value, 1)
        self.assertEqual(self.gws.resources[2].value, 2)
        self.assertEqual(self.gws.resources[0].value, 0)
        self.assertEqual(self.gws.resources[3].value, 0)


if __name__ == "__main__":
    unittest.main()
