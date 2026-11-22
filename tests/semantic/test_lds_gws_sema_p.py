from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 ds_gws_sema_p.

Models the AMD RDNA2 / GFX10 Global Wavefront Sync (GWS) hardware semaphore
P-operation (acquire / wait / decrement):
  - ds_gws_sema_p: Decrements the target GWS semaphore counter if > 0.
    If the counter is 0, suspends/blocks the calling wavefront and enqueues
    it in the GWS hardware FIFO wait queue until awakened by ds_gws_sema_v.

RDNA2 GWS Architectural Rules:
  1. GWS Resource Addressing:
     - 64 hardware resource slots (0..63) in GDS space.
     - Target resource ID: (m0 + (offset16 & 0xFFFF)) & 0x3F.
  2. Semaphore P Semantics (Dijkstra Proberen / Wait / Down):
     - If counter > 0: counter is decremented by 1 (counter -= 1); acquire succeeds immediately.
     - If counter == 0: wavefront suspends and is enqueued into the resource FIFO wait queue.
  3. Resource Isolation & Mode Enforcement:
     - Each resource slot is independently configured (UNINITIALIZED, SEMAPHORE, BARRIER).
     - Executing ds_gws_sema_p on a BARRIER resource triggers an architectural type fault.

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
    acquire_count: int = 0
    block_count: int = 0

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
        res.acquire_count = 0
        res.block_count = 0

    def wait_p(self, res_id: int, wave_id: int) -> Tuple[bool, int, bool]:
        """Executes a semaphore P (wait/decrement/acquire) operation.

        Returns:
            Tuple[bool, int, bool]: (acquired, new_counter_value, blocked)
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_p")
        res.res_type = GWSResourceType.SEMAPHORE

        if res.value > 0:
            res.value = (res.value - 1) & MASK32
            res.acquire_count += 1
            return True, res.value, False
        else:
            res.wait_queue.append(wave_id)
            res.block_count += 1
            return False, 0, True

    def signal_v(self, res_id: int) -> Tuple[int, Optional[int]]:
        """Executes a semaphore V (signal/increment/post) operation.

        Returns:
            Tuple[int, Optional[int]]: (new_counter_value, awakened_wavefront_id_or_None)
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_v")
        res.res_type = GWSResourceType.SEMAPHORE

        res.signal_count += 1
        if res.wait_queue:
            awakened = res.wait_queue.pop(0)
            return res.value, awakened
        else:
            res.value = (res.value + 1) & MASK32
            return res.value, None


def ref_gws_resource_id(m0_val: int, offset: int) -> int:
    """Calculates target GWS resource index from M0 base register and offset."""
    base = m0_val & GWS_RESOURCE_ID_MASK
    return (base + (offset & OFFSET16_MASK)) & GWS_RESOURCE_ID_MASK


def ref_ds_gws_sema_p(subsystem: GWSSubsystem, m0_val: int, offset: int, wave_id: int) -> Tuple[bool, int, bool]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_p."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.wait_p(rid, wave_id)


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_acquire_decrements_counter(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=1)
        acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=10)
        self.assertTrue(acquired)
        self.assertEqual(val, 0)
        self.assertFalse(blocked)
        self.assertEqual(self.gws.resources[rid].value, 0)
        self.assertEqual(self.gws.resources[rid].acquire_count, 1)

    def test_consecutive_acquires_until_empty(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=10)
        for expected_val in range(9, -1, -1):
            acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=1)
            self.assertTrue(acquired)
            self.assertEqual(val, expected_val)
            self.assertFalse(blocked)
        self.assertEqual(self.gws.resources[rid].value, 0)
        self.assertEqual(self.gws.resources[rid].acquire_count, 10)

    def test_acquire_on_zero_counter_blocks_wavefront(self) -> None:
        rid = 12
        self.gws.init_resource(rid, init_val=0)
        acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=42)
        self.assertFalse(acquired)
        self.assertEqual(val, 0)
        self.assertTrue(blocked)
        self.assertEqual(self.gws.resources[rid].wait_queue, [42])
        self.assertEqual(self.gws.resources[rid].block_count, 1)

    def test_fifo_queue_ordering_for_multiple_waiting_wavefronts(self) -> None:
        rid = 15
        self.gws.init_resource(rid, init_val=0)
        waiting_waves = [101, 102, 103, 104, 105]
        for w in waiting_waves:
            acquired, _, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=w)
            self.assertFalse(acquired)
            self.assertTrue(blocked)
        self.assertEqual(self.gws.resources[rid].wait_queue, waiting_waves)

        # Signals should wake wavefronts in exact FIFO order
        for expected_w in waiting_waves:
            _, awakened = self.gws.signal_v(rid)
            self.assertEqual(awakened, expected_w)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_paired_signal_v_and_wait_p(self) -> None:
        rid = 20
        # Start at 0, wave 99 blocks
        self.gws.wait_p(rid, wave_id=99)
        self.assertEqual(self.gws.resources[rid].wait_queue, [99])

        # Signal wakes wave 99, counter stays 0
        new_val, awakened = self.gws.signal_v(rid)
        self.assertEqual(awakened, 99)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.gws.resources[rid].wait_queue, [])

        # Another signal increments counter to 1
        new_val, awakened = self.gws.signal_v(rid)
        self.assertIsNone(awakened)
        self.assertEqual(new_val, 1)

        # Subsequent wait_p acquires immediately
        acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=100)
        self.assertTrue(acquired)
        self.assertEqual(val, 0)
        self.assertFalse(blocked)

    def test_resource_id_modulo_64_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)
        self.assertEqual(ref_gws_resource_id(m0_val=0, offset=0x10005), 5)

    def test_independent_gws_resource_isolation(self) -> None:
        self.gws.init_resource(1, init_val=5)
        self.gws.init_resource(2, init_val=10)

        ref_ds_gws_sema_p(self.gws, m0_val=1, offset=0, wave_id=1)
        self.assertEqual(self.gws.resources[1].value, 4)
        self.assertEqual(self.gws.resources[2].value, 10)

        ref_ds_gws_sema_p(self.gws, m0_val=2, offset=0, wave_id=2)
        self.assertEqual(self.gws.resources[1].value, 4)
        self.assertEqual(self.gws.resources[2].value, 9)

    def test_barrier_mode_rejection(self) -> None:
        rid = 8
        self.gws.init_resource(rid, init_val=0, res_type=GWSResourceType.BARRIER)
        with self.assertRaises(TypeError):
            ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=1)


if __name__ == "__main__":
    unittest.main()
