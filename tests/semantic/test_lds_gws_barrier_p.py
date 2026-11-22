from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 ds_gws_barrier_p.

Models AMD RDNA2 / GFX10 Global Wavefront Sync (GWS) hardware barrier
operation with predicate / probe evaluation:
  - ds_gws_barrier_p: Synchronizes multiple wavefronts across workgroups
    and compute units using a dedicated GWS barrier slot in Global Data Share (GDS).
    A wavefront enters the barrier and increments the barrier's arrival counter.
    When arrived_count reaches or exceeds the target wave threshold, the barrier
    is satisfied: waiting waves are released, the arrival count resets, and the
    predicate output (SCC) is asserted (SCC=1). If the barrier threshold has
    not yet been reached, the wave is pending / held and SCC=0.
  - Predicate / Probe: Evaluates barrier satisfaction status and allows
    conditional participation. If predicate=False, the barrier check behaves
    as a non-advancing probe.

RDNA2 GWS Architectural Rules:
  1. GWS Resource Addressing:
     - 64 hardware resource slots (0..63) in GDS space.
     - Effective target resource ID: (m0 + (offset16 & 0xFFFF)) & 0x3F.
  2. Barrier Synchronization Semantics:
     - Target wave count is specified via vdata0 (clamped to at least 1).
     - Each arriving active wavefront increments arrived_count.
     - If arrived_count >= target_count:
         arrived_count = 0
         total_releases += 1
         satisfied = True
         scc = 1
     - Else:
         satisfied = False
         scc = 0
  3. Resource Isolation & Mode Enforcement:
     - Each resource slot is independently configured (UNINITIALIZED, SEMAPHORE, BARRIER).
     - Executing ds_gws_barrier_p on a resource configured as SEMAPHORE raises TypeError.

Standard library Python only (python -I, struct, math, unittest). Zero game data.
"""

import enum
import math
import re
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 GWS Architectural Constants
GWS_NUM_RESOURCES: int = 64
GWS_RESOURCE_ID_MASK: int = 0x3F
OFFSET16_MASK: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
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
    """Calculate the 6-bit GWS resource identifier (0..63)."""
    return (m0 + (offset & OFFSET16_MASK)) & GWS_RESOURCE_ID_MASK


def calc_gws_effective_offset(offset: int = 0, m0: int = 0) -> int:
    """Calculate effective offset before modulo masking."""
    return (m0 + (offset & OFFSET16_MASK)) & MASK32


def ref_ds_gws_barrier_p(
    current_arrived: int,
    target_count: int,
    predicate: bool = True,
) -> Tuple[bool, int, bool]:
    """Pure mathematical reference oracle for ds_gws_barrier_p.

    Args:
        current_arrived: Current number of arrived waves in the barrier slot.
        target_count: Target wave threshold required to release the barrier.
        predicate: Predicate flag governing wave barrier participation.

    Returns:
        Tuple[bool, int, bool]: (barrier_satisfied, next_arrived_count, predicate_satisfied)
    """
    target = max(1, u32(target_count))
    curr = u32(current_arrived)

    if not predicate:
        return False, curr, False

    new_arrived = u32(curr + 1)
    if new_arrived >= target:
        return True, 0, True
    return False, new_arrived, False


def evaluate_gws_barrier_p_status(
    target_count: int,
    current_arrived: int,
    offset: int = 0,
    m0: int = 0,
    predicate: bool = True,
    exec_mask: int = 0xFFFFFFFF,
) -> dict[str, Union[bool, int]]:
    """Architectural evaluation of status flags and condition codes for ds_gws_barrier_p."""
    satisfied, next_arr, pred_sat = ref_ds_gws_barrier_p(current_arrived, target_count, predicate=predicate)
    eff_16 = (m0 & OFFSET16_MASK) + (offset & OFFSET16_MASK)
    eff_32 = m0 + (offset & OFFSET16_MASK)

    return {
        "barrier_satisfied": satisfied,
        "predicate_passed": pred_sat,
        "arrived_zero": (next_arr == 0),
        "offset_carry_out_16": (eff_16 > OFFSET16_MASK),
        "offset_carry_out_32": (eff_32 > MASK32),
        "active_wave": (exec_mask != 0),
        "scc_predicate": 1 if satisfied else 0,
        "vcc_unmodified": True,
    }


class GWSResourceType(Enum):
    """GWS hardware resource mode."""
    UNINITIALIZED = "uninitialized"
    SEMAPHORE = "semaphore"
    BARRIER = "barrier"


@dataclass
class GWSBarrierPResource:
    """Architectural state representation of an individual GWS hardware barrier slot."""
    res_id: int
    res_type: GWSResourceType = GWSResourceType.BARRIER
    target_count: int = 1
    arrived_count: int = 0
    total_releases: int = 0
    probe_pass_count: int = 0
    probe_fail_count: int = 0
    participating_waves: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0 <= self.res_id < GWS_NUM_RESOURCES):
            raise ValueError(f"GWS resource id out of bounds [0, 63]: {self.res_id}")

    def arrive(
        self,
        wave_id: int,
        target: Optional[int] = None,
        predicate: bool = True,
    ) -> Tuple[bool, int, bool]:
        """A wave executes ds_gws_barrier_p on this barrier slot."""
        if self.res_type == GWSResourceType.SEMAPHORE:
            raise TypeError(
                f"GWS resource {self.res_id} configured as SEMAPHORE, cannot execute ds_gws_barrier_p"
            )
        self.res_type = GWSResourceType.BARRIER

        if target is not None:
            self.target_count = max(1, u32(target))

        if predicate:
            self.participating_waves.append(wave_id)

        satisfied, next_count, pred_sat = ref_ds_gws_barrier_p(
            self.arrived_count, self.target_count, predicate=predicate
        )

        if satisfied:
            self.arrived_count = 0
            self.total_releases += 1
            self.probe_pass_count += 1
            self.participating_waves.clear()
            return True, 0, True
        else:
            self.arrived_count = next_count
            if predicate:
                self.probe_fail_count += 1
            return False, self.arrived_count, False

    def probe_only(self) -> Tuple[bool, int, int]:
        """Non-advancing probe of current barrier status."""
        return (self.arrived_count == 0 and self.total_releases > 0), self.arrived_count, self.target_count

    def reset(self) -> None:
        """Reset barrier slot state to initial defaults."""
        self.target_count = 1
        self.arrived_count = 0
        self.total_releases = 0
        self.probe_pass_count = 0
        self.probe_fail_count = 0
        self.participating_waves.clear()


class GWSMemory:
    """Architectural memory model representing the Global Wave Sync (GWS) subsystem."""

    def __init__(self, num_resources: int = GWS_NUM_RESOURCES) -> None:
        if num_resources <= 0 or (num_resources & (num_resources - 1)) != 0:
            raise ValueError(f"num_resources must be a positive power of two, got {num_resources}")
        self.num_resources = num_resources
        self.mask = num_resources - 1
        self.barriers: List[GWSBarrierPResource] = [
            GWSBarrierPResource(res_id=i) for i in range(num_resources)
        ]

    def reset(self) -> None:
        for b in self.barriers:
            b.reset()

    def init_resource(
        self,
        res_id: int,
        target_count: int = 1,
        res_type: GWSResourceType = GWSResourceType.BARRIER,
    ) -> None:
        rid = res_id & self.mask
        res = self.barriers[rid]
        res.res_type = res_type
        res.target_count = max(1, u32(target_count))
        res.arrived_count = 0
        res.total_releases = 0
        res.probe_pass_count = 0
        res.probe_fail_count = 0
        res.participating_waves.clear()

    def read_arrived(self, resource_id: int) -> int:
        idx = resource_id & self.mask
        return self.barriers[idx].arrived_count

    def read_releases(self, resource_id: int) -> int:
        idx = resource_id & self.mask
        return self.barriers[idx].total_releases

    def barrier_sync(
        self,
        offset: int = 0,
        m0: int = 0,
        target_count: int = 1,
        wave_id: int = 0,
        predicate: bool = True,
    ) -> Tuple[int, bool, int, bool]:
        """Execute ds_gws_barrier_p on the targeted GWS resource."""
        rid = calc_gws_resource_id(offset=offset, m0=m0)
        idx = rid & self.mask
        satisfied, arrived, pred_sat = self.barriers[idx].arrive(
            wave_id=wave_id, target=target_count, predicate=predicate
        )
        return rid, satisfied, arrived, pred_sat


class TestGWSBarrierPReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_single_wave_barrier(self) -> None:
        satisfied, arrived, pred = ref_ds_gws_barrier_p(current_arrived=0, target_count=1)
        self.assertTrue(satisfied)
        self.assertEqual(arrived, 0)
        self.assertTrue(pred)

    def test_oracle_multi_wave_barrier(self) -> None:
        # Wave 1 arrives (target 3)
        sat1, arr1, pred1 = ref_ds_gws_barrier_p(current_arrived=0, target_count=3)
        self.assertFalse(sat1)
        self.assertEqual(arr1, 1)
        self.assertFalse(pred1)

        # Wave 2 arrives
        sat2, arr2, pred2 = ref_ds_gws_barrier_p(current_arrived=arr1, target_count=3)
        self.assertFalse(sat2)
        self.assertEqual(arr2, 2)
        self.assertFalse(pred2)

        # Wave 3 arrives -> satisfied!
        sat3, arr3, pred3 = ref_ds_gws_barrier_p(current_arrived=arr2, target_count=3)
        self.assertTrue(sat3)
        self.assertEqual(arr3, 0)
        self.assertTrue(pred3)

    def test_oracle_predicate_false_bypasses_arrival(self) -> None:
        sat, arr, pred = ref_ds_gws_barrier_p(current_arrived=2, target_count=3, predicate=False)
        self.assertFalse(sat)
        self.assertEqual(arr, 2)
        self.assertFalse(pred)

    def test_oracle_zero_or_negative_target_treated_as_at_least_one(self) -> None:
        sat, arr, pred = ref_ds_gws_barrier_p(current_arrived=0, target_count=0)
        self.assertTrue(sat)
        self.assertEqual(arr, 0)
        self.assertTrue(pred)

    def test_resource_id_calculation(self) -> None:
        self.assertEqual(calc_gws_resource_id(0, 0), 0)
        self.assertEqual(calc_gws_resource_id(15, 0), 15)
        self.assertEqual(calc_gws_resource_id(63, 0), 63)
        self.assertEqual(calc_gws_resource_id(64, 0), 0)
        self.assertEqual(calc_gws_resource_id(65, 0), 1)
        self.assertEqual(calc_gws_resource_id(10, 4), 14)
        self.assertEqual(calc_gws_resource_id(60, 8), 4)

    def test_effective_offset_calculation(self) -> None:
        self.assertEqual(calc_gws_effective_offset(0x20, 0x10), 0x30)
        self.assertEqual(calc_gws_effective_offset(0xFFFF, 1), 0x10000)

    def test_gws_memory_barrier_sync(self) -> None:
        gws = GWSMemory()
        self.assertEqual(gws.num_resources, 64)

        rid1, sat1, arr1, pred1 = gws.barrier_sync(offset=5, m0=0, target_count=2, wave_id=101)
        self.assertEqual(rid1, 5)
        self.assertFalse(sat1)
        self.assertEqual(arr1, 1)
        self.assertFalse(pred1)
        self.assertEqual(gws.read_arrived(5), 1)
        self.assertEqual(gws.read_releases(5), 0)

        rid2, sat2, arr2, pred2 = gws.barrier_sync(offset=5, m0=0, target_count=2, wave_id=102)
        self.assertEqual(rid2, 5)
        self.assertTrue(sat2)
        self.assertEqual(arr2, 0)
        self.assertTrue(pred2)
        self.assertEqual(gws.read_arrived(5), 0)
        self.assertEqual(gws.read_releases(5), 1)

    def test_barrier_reset(self) -> None:
        gws = GWSMemory()
        gws.barrier_sync(offset=8, target_count=4, wave_id=1)
        self.assertEqual(gws.read_arrived(8), 1)
        gws.barriers[8].reset()
        self.assertEqual(gws.read_arrived(8), 0)
        self.assertEqual(gws.read_releases(8), 0)

    def test_mode_enforcement_rejects_semaphore(self) -> None:
        gws = GWSMemory()
        gws.init_resource(res_id=12, target_count=1, res_type=GWSResourceType.SEMAPHORE)
        with self.assertRaises(TypeError):
            gws.barrier_sync(offset=12, m0=0, target_count=1, wave_id=1)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) GWS opcodes."""
    DS_GWS_BARRIER_P = "ds_gws_barrier_p"
    DS_GWS_BARRIER = "ds_gws_barrier"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}[{self.index}] "
                f"(expected 0 <= index < {max_limit})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}[{self.index}]"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Immediate:
    """32-bit immediate constant operand."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u32(self.val))

    def __str__(self) -> str:
        return f"0x{self.val:X}"


Operand = Union[RegisterOperand, Immediate]


@dataclass(frozen=True)
class DSGWSBarrierPInstruction:
    """Symbolic AST node for ds_gws_barrier_p instruction."""
    vdata0: RegisterOperand
    offset: int = 0
    gds: bool = True
    predicate: bool = True
    opcode: DSOpcode = DSOpcode.DS_GWS_BARRIER_P

    def __post_init__(self) -> None:
        if self.vdata0.reg_type != RegType.VGPR:
            raise ValueError(f"vdata0 must be a VGPR, got {self.vdata0.reg_type}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(
                f"offset out of bounds: {self.offset} (expected 0 <= offset <= {OFFSET16_MASK})"
            )
        if not self.gds:
            raise ValueError("gds modifier is mandatory for GWS instructions")

    def disassemble(self) -> str:
        """Emit canonical AMD assembly string."""
        parts = [self.opcode.value, str(self.vdata0)]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return f"{parts[0]} {', '.join(parts[1:])}"


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        instr = DSGWSBarrierPInstruction(vdata0=v0, offset=0, gds=True)

        self.assertEqual(instr.opcode, DSOpcode.DS_GWS_BARRIER_P)
        self.assertEqual(instr.vdata0.index, 0)
        self.assertEqual(instr.offset, 0)
        self.assertTrue(instr.gds)
        self.assertTrue(instr.predicate)

    def test_valid_ast_with_offset(self) -> None:
        v5 = RegisterOperand(5, RegType.VGPR)
        instr = DSGWSBarrierPInstruction(vdata0=v5, offset=16, gds=True)

        self.assertEqual(instr.vdata0.index, 5)
        self.assertEqual(instr.offset, 16)
        self.assertTrue(instr.gds)

    def test_disassembly_formatting(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v7 = RegisterOperand(7, RegType.VGPR)

        instr_no_offset = DSGWSBarrierPInstruction(vdata0=v0, offset=0, gds=True)
        self.assertEqual(instr_no_offset.disassemble(), "ds_gws_barrier_p v[0], gds")

        instr_with_offset = DSGWSBarrierPInstruction(vdata0=v7, offset=32, gds=True)
        self.assertEqual(instr_with_offset.disassemble(), "ds_gws_barrier_p v[7], offset:32, gds")


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarState:
    """Scalar machine state comprising register files, m0, and GWS memory."""

    def __init__(self, num_gws_resources: int = GWS_NUM_RESOURCES) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.m0: int = 0
        self.vcc: int = 0
        self.scc: int = 0
        self.gws: GWSMemory = GWSMemory(num_resources=num_gws_resources)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)

    def get_m0(self) -> int:
        return self.m0

    def set_m0(self, val: int) -> None:
        self.m0 = u32(val)

    def get_reg(self, reg: RegisterOperand) -> int:
        if reg.reg_type == RegType.VGPR:
            return self.get_vgpr(reg.index)
        return self.get_sgpr(reg.index)

    def set_reg(self, reg: RegisterOperand, val: int) -> None:
        if reg.reg_type == RegType.VGPR:
            self.set_vgpr(reg.index, val)
        else:
            self.set_sgpr(reg.index, val)

    def resolve_operand(self, op: Operand) -> int:
        if isinstance(op, Immediate):
            return op.val
        return self.get_reg(op)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_gws_barrier_p."""

    def execute(
        self,
        instr: DSGWSBarrierPInstruction,
        state: ScalarState,
        wave_id: int = 0,
    ) -> Tuple[int, bool, int, bool]:
        """Execute a single ds_gws_barrier_p instruction on scalar state."""
        target_count = state.get_reg(instr.vdata0)
        m0 = state.get_m0()

        rid, satisfied, arrived, pred_sat = state.gws.barrier_sync(
            offset=instr.offset,
            m0=m0,
            target_count=target_count,
            wave_id=wave_id,
            predicate=instr.predicate,
        )

        state.scc = 1 if satisfied else 0
        return rid, satisfied, arrived, pred_sat


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_single_wave_execution(self) -> None:
        self.state.set_vgpr(0, 1)
        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=0)

        rid, sat, arr, pred = self.interp.execute(instr, self.state, wave_id=1)
        self.assertEqual(rid, 0)
        self.assertTrue(sat)
        self.assertEqual(arr, 0)
        self.assertTrue(pred)
        self.assertEqual(self.state.scc, 1)
        self.assertEqual(self.state.gws.read_releases(0), 1)

    def test_scalar_multi_wave_execution(self) -> None:
        self.state.set_vgpr(1, 2)
        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=5)

        # Wave 1 arrives
        rid1, sat1, arr1, pred1 = self.interp.execute(instr, self.state, wave_id=10)
        self.assertEqual(rid1, 5)
        self.assertFalse(sat1)
        self.assertEqual(arr1, 1)
        self.assertFalse(pred1)
        self.assertEqual(self.state.scc, 0)

        # Wave 2 arrives -> satisfied
        rid2, sat2, arr2, pred2 = self.interp.execute(instr, self.state, wave_id=11)
        self.assertEqual(rid2, 5)
        self.assertTrue(sat2)
        self.assertEqual(arr2, 0)
        self.assertTrue(pred2)
        self.assertEqual(self.state.scc, 1)
        self.assertEqual(self.state.gws.read_releases(5), 1)

    def test_scalar_predicate_false_bypasses(self) -> None:
        self.state.set_vgpr(0, 4)
        instr = DSGWSBarrierPInstruction(
            vdata0=RegisterOperand(0, RegType.VGPR), offset=7, predicate=False
        )

        rid, sat, arr, pred = self.interp.execute(instr, self.state, wave_id=1)
        self.assertEqual(rid, 7)
        self.assertFalse(sat)
        self.assertEqual(arr, 0)
        self.assertFalse(pred)
        self.assertEqual(self.state.scc, 0)
        self.assertEqual(self.state.gws.read_arrived(7), 0)

    def test_scalar_with_offset_and_m0(self) -> None:
        self.state.set_m0(8)
        self.state.set_vgpr(2, 1)
        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=4)

        rid, sat, arr, pred = self.interp.execute(instr, self.state, wave_id=20)
        self.assertEqual(rid, 12)
        self.assertTrue(sat)
        self.assertEqual(self.state.get_m0(), 8)

    def test_scalar_registers_unmodified(self) -> None:
        for i in range(DEFAULT_NUM_VGPRS):
            self.state.set_vgpr(i, 0x1100 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.state.set_sgpr(i, 0x2200 + i)
        self.state.set_m0(15)

        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(50, RegType.VGPR), offset=3)
        self.interp.execute(instr, self.state, wave_id=1)

        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(self.state.get_vgpr(i), 0x1100 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(self.state.get_sgpr(i), 0x2200 + i)
        self.assertEqual(self.state.get_m0(), 15)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDState:
    """SIMD wavefront execution state supporting Wave32 and Wave64 execution modes."""

    def __init__(
        self,
        num_lanes: int = 64,
        exec_mask: Optional[int] = None,
        num_gws_resources: int = GWS_NUM_RESOURCES,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.full_mask = (1 << num_lanes) - 1
        self.exec_mask = self.full_mask if exec_mask is None else (exec_mask & self.full_mask)
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.m0: int = 0
        self.vcc: int = 0
        self.scc: int = 0
        self.gws: GWSMemory = GWSMemory(num_resources=num_gws_resources)

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))

    def get_lane_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane][index]

    def set_lane_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)

    def get_m0(self) -> int:
        return self.m0

    def set_m0(self, val: int) -> None:
        self.m0 = u32(val)

    def find_first_active_lane(self) -> Optional[int]:
        """Find the index of the first active lane according to the EXEC mask (ctz)."""
        if self.exec_mask == 0:
            return None
        return (self.exec_mask & -self.exec_mask).bit_length() - 1


class SIMDInterpreter:
    """SIMD multi-lane instruction interpreter for ds_gws_barrier_p."""

    def execute(
        self,
        instr: DSGWSBarrierPInstruction,
        state: SIMDState,
        wave_id: int = 0,
    ) -> Optional[Tuple[int, int, bool, int, bool]]:
        """Execute ds_gws_barrier_p across SIMD lanes with EXEC masking."""
        first_lane = state.find_first_active_lane()
        if first_lane is None:
            return None

        target_count = state.get_lane_vgpr(first_lane, instr.vdata0.index)
        m0 = state.get_m0()

        rid, satisfied, arrived, pred_sat = state.gws.barrier_sync(
            offset=instr.offset,
            m0=m0,
            target_count=target_count,
            wave_id=wave_id,
            predicate=instr.predicate,
        )

        state.scc = 1 if satisfied else 0
        return first_lane, rid, satisfied, arrived, pred_sat


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 3, 1)
        for lane in range(1, 32):
            state.set_lane_vgpr(lane, 3, 10)

        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(3, RegType.VGPR), offset=0)
        res = interp.execute(instr, state, wave_id=1)

        self.assertIsNotNone(res)
        first_lane, rid, sat, arr, pred = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 0)
        self.assertTrue(sat)
        self.assertEqual(arr, 0)
        self.assertTrue(pred)
        self.assertEqual(state.scc, 1)

    def test_wave64_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 4, 2)
        for lane in range(1, 64):
            state.set_lane_vgpr(lane, 4, 99)

        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(4, RegType.VGPR), offset=10)
        res = interp.execute(instr, state, wave_id=5)

        self.assertIsNotNone(res)
        first_lane, rid, sat, arr, pred = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 10)
        self.assertFalse(sat)
        self.assertEqual(arr, 1)
        self.assertFalse(pred)
        self.assertEqual(state.scc, 0)

    def test_wave32_divergent_mask_skips_inactive_lanes(self) -> None:
        mask = 0xFFFFFFF0
        state = SIMDState(num_lanes=32, exec_mask=mask)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 1, 100)
        state.set_lane_vgpr(4, 1, 1)

        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=7)
        res = interp.execute(instr, state, wave_id=7)

        self.assertIsNotNone(res)
        first_lane, rid, sat, arr, pred = res  # type: ignore
        self.assertEqual(first_lane, 4)
        self.assertEqual(rid, 7)
        self.assertTrue(sat)
        self.assertEqual(arr, 0)
        self.assertTrue(pred)

    def test_wave64_divergent_mask_high_lanes_only(self) -> None:
        mask = 1 << 60
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(60, 2, 1)
        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=20)

        res = interp.execute(instr, state, wave_id=60)
        self.assertIsNotNone(res)
        first_lane, rid, sat, arr, pred = res  # type: ignore
        self.assertEqual(first_lane, 60)
        self.assertEqual(rid, 20)
        self.assertTrue(sat)

    def test_zero_exec_mask_no_operation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 1, 1)
        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=2)

        res = interp.execute(instr, state, wave_id=99)
        self.assertIsNone(res)
        self.assertEqual(state.gws.read_arrived(2), 0)
        self.assertEqual(state.gws.read_releases(2), 0)

    def test_registers_preserved_across_all_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            for r in range(10):
                state.set_lane_vgpr(lane, r, (lane << 16) | r)

        instr = DSGWSBarrierPInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=3)
        interp.execute(instr, state, wave_id=1)

        for lane in range(64):
            for r in range(10):
                expected = (lane << 16) | r
                self.assertEqual(state.get_lane_vgpr(lane, r), expected)


if __name__ == "__main__":
    unittest.main()
