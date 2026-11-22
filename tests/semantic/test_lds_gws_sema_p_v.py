from __future__ import annotations

"""Reference oracle and architectural model for ds_gws_sema_p and ds_gws_sema_v.

RDNA2 / GFX10 Data Share Global Wave Sync (GWS) Semaphore Operations:
Coordinates synchronization across multiple wavefronts and workgroups using
dedicated hardware semaphore counters in Global Data Share (GDS).

Instructions:
    ds_gws_sema_p [vdata0] [offset:offset16] gds   # Semaphore Acquire / Wait (Proberen)
    ds_gws_sema_v [vdata0] [offset:offset16] gds   # Semaphore Release / Signal (Verhogen)

Semantics:
    resource_id = (m0 + (offset16 & 0xFFFF)) & 0x3F
    first_active_lane = ctz(EXEC)
    if EXEC != 0:
        if P operation:
            if GWS[resource_id].counter > 0:
                GWS[resource_id].counter -= 1
                proceed = True
            else:
                stall wave (wait in GWS queue)
                proceed = False
        if V operation:
            GWS[resource_id].counter += 1
            wake waiting wave if any
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 GWS Architectural Constants
GWS_NUM_RESOURCES: int = 64         # 64 hardware GWS semaphore slots on GFX10
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
    """Calculate the 6-bit GWS resource identifier.

    The resource ID is computed by combining m0 register and 16-bit offset,
    wrapping modulo 64 (0x3F).
    """
    return (m0 + (offset & OFFSET16_MASK)) & GWS_RESOURCE_ID_MASK


def calc_gws_effective_offset(offset: int = 0, m0: int = 0) -> int:
    """Calculate effective offset before modulo masking."""
    return (m0 + (offset & OFFSET16_MASK)) & MASK32


def ref_ds_gws_sema_p(current_val: int) -> Tuple[bool, int]:
    """Pure mathematical reference oracle for ds_gws_sema_p (acquire).

    If semaphore counter > 0: decrements counter and acquires semaphore.
    If semaphore counter == 0: stalls wave and counter remains unchanged.

    Returns:
        Tuple[bool, int]: (acquired, next_counter_value)
    """
    val = u32(current_val)
    if val > 0:
        return True, u32(val - 1)
    return False, 0


def ref_ds_gws_sema_v(current_val: int) -> int:
    """Pure mathematical reference oracle for ds_gws_sema_v (release).

    Increments semaphore counter modulo 2^32.

    Returns:
        int: next_counter_value
    """
    return u32(u32(current_val) + 1)


@dataclass
class GWSSemaphoreResource:
    """Architectural state representation of an individual GWS hardware semaphore slot."""
    counter: int = 0
    waiting_waves: list[int] = field(default_factory=list)
    p_operations: int = 0
    v_operations: int = 0

    def acquire(self, wave_id: int) -> Tuple[bool, int]:
        """Execute semaphore P (acquire/probe) operation.

        Returns:
            Tuple[bool, int]: (acquired, current_counter)
        """
        self.p_operations += 1
        if self.counter > 0:
            self.counter -= 1
            return True, self.counter
        self.waiting_waves.append(wave_id)
        return False, 0

    def release(self) -> Tuple[int, Optional[int]]:
        """Execute semaphore V (release/signal) operation.

        Returns:
            Tuple[int, Optional[int]]: (new_counter, woken_wave_id_or_none)
        """
        self.v_operations += 1
        if self.waiting_waves:
            # Wake first waiting wave in FIFO queue; semaphore token passes directly
            woken_wave = self.waiting_waves.pop(0)
            return self.counter, woken_wave
        self.counter = u32(self.counter + 1)
        return self.counter, None

    def reset(self) -> None:
        """Reset semaphore state to defaults."""
        self.counter = 0
        self.waiting_waves.clear()
        self.p_operations = 0
        self.v_operations = 0


class GWSMemory:
    """Architectural memory model representing the Global Wave Sync (GWS) subsystem."""

    def __init__(self, num_resources: int = GWS_NUM_RESOURCES) -> None:
        if num_resources <= 0 or (num_resources & (num_resources - 1)) != 0:
            raise ValueError(f"num_resources must be a positive power of two, got {num_resources}")
        self.num_resources = num_resources
        self.mask = num_resources - 1
        self.semaphores: list[GWSSemaphoreResource] = [
            GWSSemaphoreResource() for _ in range(num_resources)
        ]

    def read_counter(self, resource_id: int) -> int:
        """Read the current semaphore counter."""
        idx = resource_id & self.mask
        return self.semaphores[idx].counter

    def set_counter(self, resource_id: int, val: int) -> None:
        """Directly initialize the semaphore counter."""
        idx = resource_id & self.mask
        self.semaphores[idx].counter = u32(val)

    def sema_p(
        self,
        offset: int = 0,
        m0: int = 0,
        wave_id: int = 0,
    ) -> Tuple[int, bool, int]:
        """Execute ds_gws_sema_p on targeted GWS resource.

        Returns:
            Tuple[int, bool, int]: (resource_id, acquired, current_counter)
        """
        rid = calc_gws_resource_id(offset=offset, m0=m0)
        idx = rid & self.mask
        acquired, counter = self.semaphores[idx].acquire(wave_id=wave_id)
        return rid, acquired, counter

    def sema_v(
        self,
        offset: int = 0,
        m0: int = 0,
    ) -> Tuple[int, int, Optional[int]]:
        """Execute ds_gws_sema_v on targeted GWS resource.

        Returns:
            Tuple[int, int, Optional[int]]: (resource_id, new_counter, woken_wave_id_or_none)
        """
        rid = calc_gws_resource_id(offset=offset, m0=m0)
        idx = rid & self.mask
        counter, woken_wave = self.semaphores[idx].release()
        return rid, counter, woken_wave


class TestGWSSemaphoreReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_sema_p_acquire_success(self) -> None:
        # Counter > 0 allows acquire and decrements
        acquired, val = ref_ds_gws_sema_p(current_val=5)
        self.assertTrue(acquired)
        self.assertEqual(val, 4)

        acquired2, val2 = ref_ds_gws_sema_p(current_val=1)
        self.assertTrue(acquired2)
        self.assertEqual(val2, 0)

    def test_oracle_sema_p_acquire_blocked(self) -> None:
        # Counter == 0 fails to acquire and leaves counter at 0
        acquired, val = ref_ds_gws_sema_p(current_val=0)
        self.assertFalse(acquired)
        self.assertEqual(val, 0)

    def test_oracle_sema_v_release(self) -> None:
        self.assertEqual(ref_ds_gws_sema_v(0), 1)
        self.assertEqual(ref_ds_gws_sema_v(1), 2)
        self.assertEqual(ref_ds_gws_sema_v(100), 101)
        self.assertEqual(ref_ds_gws_sema_v(0xFFFFFFFF), 0)  # 32-bit wrap

    def test_resource_id_calculation(self) -> None:
        self.assertEqual(calc_gws_resource_id(0, 0), 0)
        self.assertEqual(calc_gws_resource_id(10, 0), 10)
        self.assertEqual(calc_gws_resource_id(63, 0), 63)
        self.assertEqual(calc_gws_resource_id(64, 0), 0)
        self.assertEqual(calc_gws_resource_id(65, 0), 1)
        self.assertEqual(calc_gws_resource_id(20, 5), 25)
        self.assertEqual(calc_gws_resource_id(60, 10), 6)

    def test_effective_offset_calculation(self) -> None:
        self.assertEqual(calc_gws_effective_offset(0x10, 0x20), 0x30)
        self.assertEqual(calc_gws_effective_offset(0xFFFF, 1), 0x10000)

    def test_architectural_model_acquire_and_release(self) -> None:
        gws = GWSMemory()
        # Seed resource 3 with count 1
        gws.set_counter(3, 1)

        # First wave acquires successfully
        rid, acq1, cnt1 = gws.sema_p(offset=3, wave_id=101)
        self.assertEqual(rid, 3)
        self.assertTrue(acq1)
        self.assertEqual(cnt1, 0)
        self.assertEqual(gws.read_counter(3), 0)

        # Second wave tries to acquire -> blocked and queued
        rid2, acq2, cnt2 = gws.sema_p(offset=3, wave_id=102)
        self.assertEqual(rid2, 3)
        self.assertFalse(acq2)
        self.assertEqual(cnt2, 0)
        self.assertEqual(len(gws.semaphores[3].waiting_waves), 1)

        # Release is executed -> unblocks wave 102!
        rid_v, cnt_v, woken = gws.sema_v(offset=3)
        self.assertEqual(rid_v, 3)
        self.assertEqual(woken, 102)
        self.assertEqual(len(gws.semaphores[3].waiting_waves), 0)

    def test_semaphore_reset(self) -> None:
        gws = GWSMemory()
        gws.set_counter(1, 10)
        gws.semaphores[1].waiting_waves.append(99)
        gws.semaphores[1].reset()
        self.assertEqual(gws.read_counter(1), 0)
        self.assertEqual(len(gws.semaphores[1].waiting_waves), 0)
        self.assertEqual(gws.semaphores[1].p_operations, 0)
        self.assertEqual(gws.semaphores[1].v_operations, 0)


if __name__ == "__main__":
    unittest.main()
