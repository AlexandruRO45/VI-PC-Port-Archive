from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 ds_gws_sema_br_p.

Models AMD RDNA2 / GFX10 Global Wavefront Sync (GWS) hardware semaphore
branch-probe operation (non-blocking try-acquire / probe-and-branch):
  - ds_gws_sema_br_p: Probes the targeted GWS semaphore resource counter.
    If counter > 0 (or >= requested permits), decrements counter by requested
    amount (default 1) and signals successful probe (branch taken / SCC=1).
    If counter == 0 (or < requested permits), the probe fails (branch not taken / SCC=0).
    CRITICALLY: unlike blocking ds_gws_sema_p, ds_gws_sema_br_p NEVER blocks
    or enqueues the wavefront! It is a non-blocking speculative probe.

RDNA2 GWS Architectural Rules:
  1. GWS Resource Addressing:
     - 64 hardware resource slots (0..63) in GDS space.
     - Effective target resource ID: (m0 + (offset16 & 0xFFFF)) & 0x3F.
  2. Semaphore Branch Probe Semantics:
     - Probe Condition: Checks if counter >= required_permits (default 1).
     - Success Path: Counter is atomically decremented: counter -= required_permits.
       Probe returns (success=True, new_counter=counter, branch_taken=True).
     - Failure Path: Counter is unchanged. Probe returns (success=False, new_counter=counter, branch_taken=False).
     - Non-Blocking Guarantee: The calling wavefront never halts, suspends, or sleeps.
  3. Resource Isolation & Mode Enforcement:
     - Each resource slot is independently configured (UNINITIALIZED, SEMAPHORE, BARRIER).
     - Executing ds_gws_sema_br_p on a BARRIER resource triggers an architectural type fault.

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
    probe_success_count: int = 0
    probe_fail_count: int = 0

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
        res.probe_success_count = 0
        res.probe_fail_count = 0

    def probe_p(self, res_id: int, permits: int = 1) -> Tuple[bool, int, bool]:
        """Executes a semaphore branch probe (try-acquire) operation.

        Returns:
            Tuple[bool, int, bool]: (probe_success, new_counter_value, branch_taken)
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_br_p")
        res.res_type = GWSResourceType.SEMAPHORE

        req_permits = max(1, permits & MASK32)
        if res.value >= req_permits:
            res.value = (res.value - req_permits) & MASK32
            res.probe_success_count += 1
            return True, res.value, True
        else:
            res.probe_fail_count += 1
            return False, res.value, False

    def wait_p(self, res_id: int, wave_id: int) -> Tuple[bool, int, bool]:
        """Executes a standard blocking semaphore P operation."""
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
        """Executes a semaphore V (signal/release) operation."""
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


def ref_ds_gws_sema_br_p(subsystem: GWSSubsystem, m0_val: int, offset: int, permits: int = 1) -> Tuple[bool, int, bool]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_br_p."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.probe_p(rid, permits=permits)


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_br_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_probe_success(self) -> None:
        rid = 3
        self.gws.init_resource(rid, init_val=1)
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(success)
        self.assertEqual(val, 0)
        self.assertTrue(branch)
        self.assertEqual(self.gws.resources[rid].value, 0)
        self.assertEqual(self.gws.resources[rid].probe_success_count, 1)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_single_probe_fail_when_zero_never_blocks(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=0)
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertFalse(success)
        self.assertEqual(val, 0)
        self.assertFalse(branch)
        # CRITICAL: wait_queue must be empty! Probe never suspends the calling wave
        self.assertEqual(self.gws.resources[rid].wait_queue, [])
        self.assertEqual(self.gws.resources[rid].probe_fail_count, 1)

    def test_multi_permit_probe_success(self) -> None:
        rid = 8
        self.gws.init_resource(rid, init_val=10)
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=4)
        self.assertTrue(success)
        self.assertEqual(val, 6)
        self.assertTrue(branch)
        self.assertEqual(self.gws.resources[rid].value, 6)

    def test_multi_permit_probe_fail_insufficient(self) -> None:
        rid = 9
        self.gws.init_resource(rid, init_val=3)
        # Requesting 4 permits when only 3 available
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=4)
        self.assertFalse(success)
        self.assertEqual(val, 3)  # counter is NOT decremented
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[rid].value, 3)

    def test_consecutive_probes_until_exhaustion(self) -> None:
        rid = 12
        self.gws.init_resource(rid, init_val=3)

        # Probe 1: succeeds (3 -> 2)
        s1, v1, b1 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s1)
        self.assertEqual(v1, 2)
        self.assertTrue(b1)

        # Probe 2: succeeds (2 -> 1)
        s2, v2, b2 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s2)
        self.assertEqual(v2, 1)
        self.assertTrue(b2)

        # Probe 3: succeeds (1 -> 0)
        s3, v3, b3 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s3)
        self.assertEqual(v3, 0)
        self.assertTrue(b3)

        # Probe 4: fails (0 remaining)
        s4, v4, b4 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertFalse(s4)
        self.assertEqual(v4, 0)
        self.assertFalse(b4)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_interleaved_probe_and_signal_v(self) -> None:
        rid = 16
        self.gws.init_resource(rid, init_val=0)

        # Initially fails
        s1, v1, b1 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertFalse(s1)

        # Signal V adds 1 permit
        self.gws.signal_v(rid)
        self.assertEqual(self.gws.resources[rid].value, 1)

        # Now probe succeeds
        s2, v2, b2 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s2)
        self.assertEqual(v2, 0)
        self.assertTrue(b2)

    def test_resource_id_modulo_64_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)
        self.assertEqual(ref_gws_resource_id(m0_val=0, offset=0x10005), 5)

    def test_resource_isolation(self) -> None:
        self.gws.init_resource(2, init_val=5)
        self.gws.init_resource(3, init_val=10)

        ref_ds_gws_sema_br_p(self.gws, m0_val=2, offset=0, permits=1)
        self.assertEqual(self.gws.resources[2].value, 4)
        self.assertEqual(self.gws.resources[3].value, 10)

        ref_ds_gws_sema_br_p(self.gws, m0_val=3, offset=0, permits=2)
        self.assertEqual(self.gws.resources[2].value, 4)
        self.assertEqual(self.gws.resources[3].value, 8)

    def test_barrier_mode_rejection(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=5, res_type=GWSResourceType.BARRIER)
        with self.assertRaises(TypeError):
            ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)


if __name__ == "__main__":
    unittest.main()
