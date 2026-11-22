from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_gws_barrier.

RDNA2 / GFX10 Data Share Global Wave Sync Barrier (ds_gws_barrier):
Synchronizes multiple wavefronts across workgroups and compute units using
a dedicated Global Wave Sync (GWS) hardware barrier in Global Data Share (GDS).

Syntax:
    ds_gws_barrier vdata0 [offset:offset16] gds

Semantics:
    resource_id = (m0 + (offset16 & 0xFFFF)) & 0x3F
    first_active_lane = ctz(EXEC)
    if EXEC != 0:
        target_count = VGPR[first_active_lane][vdata0] & 0xFFFFFFFF
        GWS[resource_id].arrived += 1
        if GWS[resource_id].arrived >= target_count:
            GWS[resource_id].arrived = 0
            GWS[resource_id].releases += 1
            barrier_satisfied = True
        else:
            barrier_satisfied = False
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 GWS Architectural Constants
GWS_NUM_RESOURCES: int = 64         # 64 hardware GWS barrier/resource slots on GFX10
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


def evaluate_gws_barrier_status(
    target_count: int,
    current_arrived: int,
    offset: int = 0,
    m0: int = 0,
    exec_mask: int = 0xFFFFFFFF,
) -> dict[str, Union[bool, int]]:
    """Architectural evaluation of status flags and condition codes for ds_gws_barrier."""
    satisfied, next_arr = ref_ds_gws_barrier(current_arrived, target_count)
    eff_16 = (m0 & OFFSET16_MASK) + (offset & OFFSET16_MASK)
    eff_32 = m0 + (offset & OFFSET16_MASK)

    return {
        "barrier_satisfied": satisfied,
        "arrived_zero": (next_arr == 0),
        "offset_carry_out_16": (eff_16 > OFFSET16_MASK),
        "offset_carry_out_32": (eff_32 > MASK32),
        "active_wave": (exec_mask != 0),
        "vcc_unmodified": True,
        "scc_unmodified": True,
    }


def ref_ds_gws_barrier(current_arrived: int, target_count: int) -> Tuple[bool, int]:
    """Pure mathematical reference oracle for ds_gws_barrier.

    A barrier requires target_count waves to arrive before it resolves.
    When a wave enters:
        arrived = (current_arrived + 1) & 0xFFFFFFFF
        satisfied = (arrived >= target_count)
        next_arrived = 0 if satisfied else arrived

    Returns:
        Tuple[bool, int]: (barrier_satisfied, next_arrived_count)
    """
    target = max(1, u32(target_count))
    curr = u32(current_arrived)
    new_arrived = u32(curr + 1)
    if new_arrived >= target:
        return True, 0
    return False, new_arrived


@dataclass
class GWSBarrierResource:
    """Architectural state representation of an individual GWS hardware barrier slot."""
    target_count: int = 1
    arrived_count: int = 0
    total_releases: int = 0
    participating_waves: list[int] = field(default_factory=list)

    def arrive(self, wave_id: int, target: Optional[int] = None) -> Tuple[bool, int]:
        """A wave arrives at the GWS barrier slot.

        Returns:
            Tuple[bool, int]: (satisfied, current_arrived_count)
        """
        if target is not None:
            self.target_count = max(1, u32(target))

        self.participating_waves.append(wave_id)
        satisfied, next_count = ref_ds_gws_barrier(self.arrived_count, self.target_count)
        if satisfied:
            self.arrived_count = 0
            self.total_releases += 1
            self.participating_waves.clear()
            return True, 0
        self.arrived_count = next_count
        return False, self.arrived_count

    def reset(self) -> None:
        """Reset barrier state to defaults."""
        self.target_count = 1
        self.arrived_count = 0
        self.total_releases = 0
        self.participating_waves.clear()


class GWSMemory:
    """Architectural memory model representing the Global Wave Sync (GWS) subsystem."""

    def __init__(self, num_resources: int = GWS_NUM_RESOURCES) -> None:
        if num_resources <= 0 or (num_resources & (num_resources - 1)) != 0:
            raise ValueError(f"num_resources must be a positive power of two, got {num_resources}")
        self.num_resources = num_resources
        self.mask = num_resources - 1
        self.barriers: list[GWSBarrierResource] = [GWSBarrierResource() for _ in range(num_resources)]

    def read_arrived(self, resource_id: int) -> int:
        """Read the arrived count from a GWS barrier slot."""
        idx = resource_id & self.mask
        return self.barriers[idx].arrived_count

    def read_releases(self, resource_id: int) -> int:
        """Read the total release count from a GWS barrier slot."""
        idx = resource_id & self.mask
        return self.barriers[idx].total_releases

    def barrier_sync(
        self,
        offset: int = 0,
        m0: int = 0,
        target_count: int = 1,
        wave_id: int = 0,
    ) -> Tuple[int, bool, int]:
        """Execute ds_gws_barrier on the targeted GWS resource.

        Returns:
            Tuple[int, bool, int]: (resource_id, satisfied, next_arrived_count)
        """
        rid = calc_gws_resource_id(offset=offset, m0=m0)
        idx = rid & self.mask
        satisfied, arrived = self.barriers[idx].arrive(wave_id=wave_id, target=target_count)
        return rid, satisfied, arrived


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) GWS opcodes."""
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
class DSGWSBarrierInstruction:
    """Symbolic AST node for ds_gws_barrier instruction."""
    vdata0: RegisterOperand
    offset: int = 0
    gds: bool = True
    opcode: DSOpcode = DSOpcode.DS_GWS_BARRIER

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


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarState:
    """Scalar machine state comprising register files, m0, and GWS memory."""

    def __init__(self, num_gws_resources: int = GWS_NUM_RESOURCES) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
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
    """Deterministic scalar instruction interpreter for ds_gws_barrier."""

    def execute(
        self,
        instr: DSGWSBarrierInstruction,
        state: ScalarState,
        wave_id: int = 0,
    ) -> Tuple[int, bool, int]:
        """Execute a single ds_gws_barrier instruction on scalar state.

        Returns:
            Tuple[int, bool, int]: (resource_id, satisfied, next_arrived_count)
        """
        # 1. Latch inputs prior to execution
        target_count = state.get_reg(instr.vdata0)
        m0 = state.get_m0()

        # 2. Perform GWS barrier sync
        rid, satisfied, arrived = state.gws.barrier_sync(
            offset=instr.offset,
            m0=m0,
            target_count=target_count,
            wave_id=wave_id,
        )

        # ds_gws_barrier produces no register writeback; registers are preserved
        return rid, satisfied, arrived


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
        self.vgprs: list[list[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
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
    """SIMD multi-lane instruction interpreter for ds_gws_barrier."""

    def execute(
        self,
        instr: DSGWSBarrierInstruction,
        state: SIMDState,
        wave_id: int = 0,
    ) -> Optional[Tuple[int, int, bool, int]]:
        """Execute ds_gws_barrier across SIMD lanes with EXEC masking.

        In RDNA2 hardware, ds_gws_barrier operates on the first active thread in the wavefront:
        first_lane = find_first_valid(EXEC mask).
        Target count is read from vdata0 in that lane.

        Returns:
            Optional[Tuple[int, int, bool, int]]: (first_lane, resource_id, satisfied, arrived_count),
            or None if the wave is inactive (EXEC == 0).
        """
        first_lane = state.find_first_active_lane()
        if first_lane is None:
            return None

        # Latch vdata0 from the first active lane and m0 from scalar state
        target_count = state.get_lane_vgpr(first_lane, instr.vdata0.index)
        m0 = state.get_m0()

        # Execute barrier arrival
        rid, satisfied, arrived = state.gws.barrier_sync(
            offset=instr.offset,
            m0=m0,
            target_count=target_count,
            wave_id=wave_id,
        )

        return first_lane, rid, satisfied, arrived


class TestGWSBarrierReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_single_wave_barrier(self) -> None:
        satisfied, arrived = ref_ds_gws_barrier(current_arrived=0, target_count=1)
        self.assertTrue(satisfied)
        self.assertEqual(arrived, 0)

    def test_oracle_multi_wave_barrier(self) -> None:
        # Target count 3
        # Wave 1 arrives
        sat1, arr1 = ref_ds_gws_barrier(current_arrived=0, target_count=3)
        self.assertFalse(sat1)
        self.assertEqual(arr1, 1)

        # Wave 2 arrives
        sat2, arr2 = ref_ds_gws_barrier(current_arrived=arr1, target_count=3)
        self.assertFalse(sat2)
        self.assertEqual(arr2, 2)

        # Wave 3 arrives -> satisfied!
        sat3, arr3 = ref_ds_gws_barrier(current_arrived=arr2, target_count=3)
        self.assertTrue(sat3)
        self.assertEqual(arr3, 0)

    def test_oracle_zero_or_negative_target_treated_as_at_least_one(self) -> None:
        satisfied, arrived = ref_ds_gws_barrier(current_arrived=0, target_count=0)
        self.assertTrue(satisfied)
        self.assertEqual(arrived, 0)

    def test_resource_id_calculation(self) -> None:
        self.assertEqual(calc_gws_resource_id(0, 0), 0)
        self.assertEqual(calc_gws_resource_id(15, 0), 15)
        self.assertEqual(calc_gws_resource_id(63, 0), 63)
        self.assertEqual(calc_gws_resource_id(64, 0), 0)
        self.assertEqual(calc_gws_resource_id(65, 0), 1)

        # m0 offset combined with immediate offset
        self.assertEqual(calc_gws_resource_id(10, 4), 14)
        self.assertEqual(calc_gws_resource_id(60, 8), 4)

    def test_effective_offset_calculation(self) -> None:
        self.assertEqual(calc_gws_effective_offset(0x20, 0x10), 0x30)
        self.assertEqual(calc_gws_effective_offset(0xFFFF, 1), 0x10000)

    def test_gws_memory_barrier_sync(self) -> None:
        gws = GWSMemory()
        self.assertEqual(gws.num_resources, 64)

        # Target 2 waves on resource 5
        rid1, sat1, arr1 = gws.barrier_sync(offset=5, m0=0, target_count=2, wave_id=101)
        self.assertEqual(rid1, 5)
        self.assertFalse(sat1)
        self.assertEqual(arr1, 1)
        self.assertEqual(gws.read_arrived(5), 1)
        self.assertEqual(gws.read_releases(5), 0)

        # Other resources unchanged
        self.assertEqual(gws.read_arrived(4), 0)
        self.assertEqual(gws.read_arrived(6), 0)

        # Second wave arrives on resource 5 -> barrier resolves!
        rid2, sat2, arr2 = gws.barrier_sync(offset=5, m0=0, target_count=2, wave_id=102)
        self.assertEqual(rid2, 5)
        self.assertTrue(sat2)
        self.assertEqual(arr2, 0)
        self.assertEqual(gws.read_arrived(5), 0)
        self.assertEqual(gws.read_releases(5), 1)

    def test_barrier_reset(self) -> None:
        gws = GWSMemory()
        gws.barrier_sync(offset=8, target_count=4, wave_id=1)
        self.assertEqual(gws.read_arrived(8), 1)
        gws.barriers[8].reset()
        self.assertEqual(gws.read_arrived(8), 0)
        self.assertEqual(gws.read_releases(8), 0)


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        instr = DSGWSBarrierInstruction(vdata0=v0, offset=0, gds=True)

        self.assertEqual(instr.opcode, DSOpcode.DS_GWS_BARRIER)
        self.assertEqual(instr.vdata0.index, 0)
        self.assertEqual(instr.offset, 0)
        self.assertTrue(instr.gds)

    def test_valid_ast_with_offset(self) -> None:
        v5 = RegisterOperand(5, RegType.VGPR)
        instr = DSGWSBarrierInstruction(vdata0=v5, offset=16, gds=True)

        self.assertEqual(instr.vdata0.index, 5)
        self.assertEqual(instr.offset, 16)
        self.assertTrue(instr.gds)

    def test_disassembly_formatting(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v7 = RegisterOperand(7, RegType.VGPR)

        instr_no_offset = DSGWSBarrierInstruction(vdata0=v0, offset=0, gds=True)
        self.assertEqual(instr_no_offset.disassemble(), "ds_gws_barrier v[0], gds")

        instr_with_offset = DSGWSBarrierInstruction(vdata0=v7, offset=32, gds=True)
        self.assertEqual(instr_with_offset.disassemble(), "ds_gws_barrier v[7], offset:32, gds")

    def test_rejection_sgpr_vdata0(self) -> None:
        s0 = RegisterOperand(0, RegType.SGPR)
        with self.assertRaises(ValueError):
            DSGWSBarrierInstruction(vdata0=s0, offset=0, gds=True)

    def test_rejection_out_of_bounds_offset(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            DSGWSBarrierInstruction(vdata0=v0, offset=-1, gds=True)

        with self.assertRaises(ValueError):
            DSGWSBarrierInstruction(vdata0=v0, offset=65536, gds=True)

    def test_rejection_missing_gds_modifier(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            DSGWSBarrierInstruction(vdata0=v0, offset=0, gds=False)

    def test_register_operand_validation(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_single_wave_barrier(self) -> None:
        # Target count 1: barrier resolves immediately on arrival
        self.state.set_vgpr(0, 1)
        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=0)

        rid, satisfied, arrived = self.interp.execute(instr, self.state, wave_id=1)
        self.assertEqual(rid, 0)
        self.assertTrue(satisfied)
        self.assertEqual(arrived, 0)
        self.assertEqual(self.state.gws.read_arrived(0), 0)
        self.assertEqual(self.state.gws.read_releases(0), 1)

    def test_scalar_multi_wave_barrier(self) -> None:
        # Target count 2
        self.state.set_vgpr(1, 2)
        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=5)

        # First wave arrives
        rid1, sat1, arr1 = self.interp.execute(instr, self.state, wave_id=10)
        self.assertEqual(rid1, 5)
        self.assertFalse(sat1)
        self.assertEqual(arr1, 1)
        self.assertEqual(self.state.gws.read_arrived(5), 1)
        self.assertEqual(self.state.gws.read_releases(5), 0)

        # Second wave arrives -> satisfied!
        rid2, sat2, arr2 = self.interp.execute(instr, self.state, wave_id=11)
        self.assertEqual(rid2, 5)
        self.assertTrue(sat2)
        self.assertEqual(arr2, 0)
        self.assertEqual(self.state.gws.read_arrived(5), 0)
        self.assertEqual(self.state.gws.read_releases(5), 1)

    def test_scalar_with_offset_and_m0(self) -> None:
        self.state.set_m0(8)
        self.state.set_vgpr(2, 1)
        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=4)

        rid, sat, arr = self.interp.execute(instr, self.state, wave_id=20)
        self.assertEqual(rid, 12)
        self.assertTrue(sat)
        self.assertEqual(self.state.get_m0(), 8)

    def test_scalar_registers_unmodified(self) -> None:
        for i in range(DEFAULT_NUM_VGPRS):
            self.state.set_vgpr(i, 0x1100 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.state.set_sgpr(i, 0x2200 + i)
        self.state.set_m0(15)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(50, RegType.VGPR), offset=3)
        self.interp.execute(instr, self.state, wave_id=1)

        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(self.state.get_vgpr(i), 0x1100 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(self.state.get_sgpr(i), 0x2200 + i)
        self.assertEqual(self.state.get_m0(), 15)

    def test_scalar_independent_barriers(self) -> None:
        # Resource 1 target 2, Resource 2 target 3
        self.state.set_vgpr(1, 2)
        self.state.set_vgpr(2, 3)

        instr1 = DSGWSBarrierInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=1)
        instr2 = DSGWSBarrierInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=2)

        # Wave arrives at res 1
        _, sat1, _ = self.interp.execute(instr1, self.state, wave_id=1)
        self.assertFalse(sat1)
        self.assertEqual(self.state.gws.read_arrived(1), 1)
        self.assertEqual(self.state.gws.read_arrived(2), 0)

        # Wave arrives at res 2
        _, sat2, _ = self.interp.execute(instr2, self.state, wave_id=2)
        self.assertFalse(sat2)
        self.assertEqual(self.state.gws.read_arrived(1), 1)
        self.assertEqual(self.state.gws.read_arrived(2), 1)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        # Lane 0 specifies target count 1, other lanes specify target count 10
        state.set_lane_vgpr(0, 3, 1)
        for lane in range(1, 32):
            state.set_lane_vgpr(lane, 3, 10)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(3, RegType.VGPR), offset=0)
        res = interp.execute(instr, state, wave_id=1)

        self.assertIsNotNone(res)
        first_lane, rid, sat, arr = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 0)
        # Because lane 0 specified target count 1, barrier resolves immediately!
        self.assertTrue(sat)
        self.assertEqual(arr, 0)

    def test_wave64_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 4, 2)
        for lane in range(1, 64):
            state.set_lane_vgpr(lane, 4, 99)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(4, RegType.VGPR), offset=10)
        res = interp.execute(instr, state, wave_id=5)

        self.assertIsNotNone(res)
        first_lane, rid, sat, arr = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 10)
        self.assertFalse(sat)
        self.assertEqual(arr, 1)

    def test_wave32_divergent_mask_skips_inactive_lanes(self) -> None:
        # Lanes 0..3 inactive, lane 4 active (mask: 0xFFFFFFF0)
        mask = 0xFFFFFFF0
        state = SIMDState(num_lanes=32, exec_mask=mask)
        interp = SIMDInterpreter()

        # Inactive lane 0 specifies target 100, active lane 4 specifies target 1
        state.set_lane_vgpr(0, 1, 100)
        state.set_lane_vgpr(4, 1, 1)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=7)
        res = interp.execute(instr, state, wave_id=7)

        self.assertIsNotNone(res)
        first_lane, rid, sat, arr = res  # type: ignore
        self.assertEqual(first_lane, 4)
        self.assertEqual(rid, 7)
        self.assertTrue(sat)
        self.assertEqual(arr, 0)

    def test_wave64_divergent_mask_high_lanes_only(self) -> None:
        mask = 1 << 60  # Only lane 60 active
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(60, 2, 1)
        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=20)

        res = interp.execute(instr, state, wave_id=60)
        self.assertIsNotNone(res)
        first_lane, rid, sat, arr = res  # type: ignore
        self.assertEqual(first_lane, 60)
        self.assertEqual(rid, 20)
        self.assertTrue(sat)

    def test_zero_exec_mask_no_operation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 1, 1)
        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=2)

        res = interp.execute(instr, state, wave_id=99)
        self.assertIsNone(res)
        # Barrier resource 2 must be completely untouched
        self.assertEqual(state.gws.read_arrived(2), 0)
        self.assertEqual(state.gws.read_releases(2), 0)

    def test_registers_preserved_across_all_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            for r in range(10):
                state.set_lane_vgpr(lane, r, (lane << 16) | r)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=3)
        interp.execute(instr, state, wave_id=1)

        for lane in range(64):
            for r in range(10):
                expected = (lane << 16) | r
                self.assertEqual(state.get_lane_vgpr(lane, r), expected)


class TestRegisterAliasingAndOperandPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_scalar_operand_preservation_boundary_registers(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, 0x12345678)
        state.set_vgpr(255, 0x87654321)
        state.set_m0(5)

        instr_v0 = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=1)
        rid0, sat0, arr0 = interp.execute(instr_v0, state, wave_id=1)
        self.assertEqual(rid0, 6)
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        self.assertEqual(state.get_m0(), 5)

        instr_v255 = DSGWSBarrierInstruction(vdata0=RegisterOperand(255, RegType.VGPR), offset=2)
        rid255, sat255, arr255 = interp.execute(instr_v255, state, wave_id=2)
        self.assertEqual(rid255, 7)
        self.assertEqual(state.get_vgpr(255), 0x87654321)
        self.assertEqual(state.get_m0(), 5)

    def test_scalar_full_register_file_undisturbed(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        for i in range(DEFAULT_NUM_VGPRS):
            state.set_vgpr(i, 0xA0000000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            state.set_sgpr(i, 0xB0000000 | i)
        state.set_m0(0x33)

        target_reg = 100
        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(target_reg, RegType.VGPR), offset=10)
        interp.execute(instr, state, wave_id=1)

        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(state.get_vgpr(i), 0xA0000000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(state.get_sgpr(i), 0xB0000000 | i)
        self.assertEqual(state.get_m0(), 0x33)

    def test_scalar_repeated_operand_use_across_barriers(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        target_count = 4
        state.set_vgpr(12, target_count)

        # 4 consecutive barrier arrivals using the exact same register
        for wave in range(1, 5):
            instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(12, RegType.VGPR), offset=3)
            rid, satisfied, arrived = interp.execute(instr, state, wave_id=wave)
            self.assertEqual(rid, 3)
            self.assertEqual(state.get_vgpr(12), target_count)
            if wave < 4:
                self.assertFalse(satisfied)
                self.assertEqual(arrived, wave)
            else:
                self.assertTrue(satisfied)
                self.assertEqual(arrived, 0)

        self.assertEqual(state.gws.read_releases(3), 1)

    def test_simd_multi_lane_preservation_under_divergence(self) -> None:
        mask = 0xA5A5A5A5A5A5A5A5
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            for r in range(16):
                state.set_lane_vgpr(lane, r, ((lane + 1) * 500) + r)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(5, RegType.VGPR), offset=15)
        res = interp.execute(instr, state, wave_id=10)
        self.assertIsNotNone(res)

        for lane in range(64):
            for r in range(16):
                expected = ((lane + 1) * 500) + r
                self.assertEqual(state.get_lane_vgpr(lane, r), expected)


class TestConditionCodesStatusFlagsAndCarryBorrow(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_preserved_scalar(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vcc = 0x11223344_55667788
        state.scc = 1
        state.set_vgpr(0, 1)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=0)
        interp.execute(instr, state, wave_id=1)

        self.assertEqual(state.vcc, 0x11223344_55667788)
        self.assertEqual(state.scc, 1)

    def test_vcc_and_scc_preserved_simd(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        state.vcc = 0xCAFEBABE
        state.scc = 0
        state.set_lane_vgpr(0, 1, 1)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=2)
        interp.execute(instr, state, wave_id=1)

        self.assertEqual(state.vcc, 0xCAFEBABE)
        self.assertEqual(state.scc, 0)

    def test_no_carry_borrow_generated(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vcc = 0
        state.scc = 0
        state.set_vgpr(0, 0xFFFFFFFF)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=63)
        interp.execute(instr, state, wave_id=1)

        self.assertEqual(state.vcc, 0)
        self.assertEqual(state.scc, 0)

    def test_status_flags_barrier_satisfied_and_zero(self) -> None:
        # Target count 2, currently 0 -> next is 1, not satisfied, not arrived_zero
        st1 = evaluate_gws_barrier_status(target_count=2, current_arrived=0)
        self.assertFalse(st1["barrier_satisfied"])
        self.assertFalse(st1["arrived_zero"])

        # Target count 2, currently 1 -> next is 2, satisfied, arrived_zero=True
        st2 = evaluate_gws_barrier_status(target_count=2, current_arrived=1)
        self.assertTrue(st2["barrier_satisfied"])
        self.assertTrue(st2["arrived_zero"])

    def test_status_flags_offset_carry_out(self) -> None:
        st_no_carry = evaluate_gws_barrier_status(target_count=1, current_arrived=0, offset=10, m0=20)
        self.assertFalse(st_no_carry["offset_carry_out_16"])
        self.assertFalse(st_no_carry["offset_carry_out_32"])

        st_carry_16 = evaluate_gws_barrier_status(target_count=1, current_arrived=0, offset=0x0020, m0=0xFFF0)
        self.assertTrue(st_carry_16["offset_carry_out_16"])
        self.assertFalse(st_carry_16["offset_carry_out_32"])

        st_carry_32 = evaluate_gws_barrier_status(target_count=1, current_arrived=0, offset=1, m0=0xFFFFFFFF)
        self.assertTrue(st_carry_32["offset_carry_out_32"])

    def test_status_flags_wave_activity(self) -> None:
        st_active = evaluate_gws_barrier_status(target_count=1, current_arrived=0, exec_mask=0x1)
        self.assertTrue(st_active["active_wave"])

        st_inactive = evaluate_gws_barrier_status(target_count=1, current_arrived=0, exec_mask=0x0)
        self.assertFalse(st_inactive["active_wave"])


class TestIEEE754BoundaryExtremesAndNaNHandling(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    # Canonical IEEE 754 float32 bit representations
    F32_POS_ZERO: int = 0x0000_0000
    F32_NEG_ZERO: int = 0x8000_0000
    F32_MIN_SUBNORMAL: int = 0x0000_0001
    F32_MAX_SUBNORMAL: int = 0x007F_FFFF
    F32_NEG_MIN_SUBNORMAL: int = 0x8000_0001
    F32_NEG_MAX_SUBNORMAL: int = 0x807F_FFFF
    F32_MIN_NORMAL: int = 0x0080_0000
    F32_ONE: int = 0x3F80_0000
    F32_NEG_ONE: int = 0xBF80_0000
    F32_MAX_NORMAL: int = 0x7F7F_FFFF
    F32_NEG_MAX_NORMAL: int = 0xFF7F_FFFF
    F32_POS_INF: int = 0x7F80_0000
    F32_NEG_INF: int = 0xFF80_0000
    F32_QNAN_CANONICAL: int = 0x7FC0_0000
    F32_SNAN: int = 0x7F80_0001
    F32_NEG_QNAN: int = 0xFFC0_0000
    F32_CUSTOM_NAN_PAYLOAD: int = 0x7FA5_BEEF

    # Packed 16-bit float patterns
    F16_PACKED_ONES: int = 0x3C00_3C00
    F16_PACKED_NANS: int = 0x7E00_7E00

    def test_signed_zeros_target_count(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        # +0.0 (0x00000000): target count 0 clamps to minimum 1, satisfies on first arrival
        state.set_vgpr(0, self.F32_POS_ZERO)
        instr0 = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=0)
        rid0, sat0, arr0 = interp.execute(instr0, state, wave_id=1)
        self.assertTrue(sat0)
        self.assertEqual(state.get_vgpr(0), self.F32_POS_ZERO)

        # -0.0 (0x80000000): bit 31 is set, interpreted as large unsigned integer (2^31)
        state.set_vgpr(1, self.F32_NEG_ZERO)
        instr1 = DSGWSBarrierInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=1)
        rid1, sat1, arr1 = interp.execute(instr1, state, wave_id=2)
        self.assertFalse(sat1)
        self.assertEqual(arr1, 1)
        self.assertEqual(state.get_vgpr(1), self.F32_NEG_ZERO)

    def test_subnormals_preservation_no_ftz(self) -> None:
        # F32_MIN_SUBNORMAL is 0x00000001 -> target 1 wave -> resolves immediately
        state = ScalarState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, self.F32_MIN_SUBNORMAL)
        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=2)
        rid, sat, arr = interp.execute(instr, state, wave_id=1)
        self.assertTrue(sat)
        self.assertEqual(arr, 0)
        # Register must retain subnormal bit pattern without Flush-To-Zero
        self.assertEqual(state.get_vgpr(0), self.F32_MIN_SUBNORMAL)

    def test_normals_and_infinities_preservation(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        patterns = [
            (self.F32_MIN_NORMAL, 3),
            (self.F32_MAX_NORMAL, 4),
            (self.F32_POS_INF, 5),
            (self.F32_NEG_INF, 6),
        ]

        for pattern, rid in patterns:
            state.set_vgpr(0, pattern)
            instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=rid)
            r, sat, arr = interp.execute(instr, state, wave_id=1)
            self.assertEqual(r, rid)
            self.assertEqual(state.get_vgpr(0), pattern)

    def test_nans_payload_preservation_no_quieting(self) -> None:
        # Signaling NaNs must NOT be quieted
        state = ScalarState()
        interp = ScalarInterpreter()

        nan_patterns = [
            (self.F32_QNAN_CANONICAL, 10),
            (self.F32_SNAN, 11),
            (self.F32_NEG_QNAN, 12),
            (self.F32_CUSTOM_NAN_PAYLOAD, 13),
        ]

        for pattern, rid in nan_patterns:
            state.set_vgpr(0, pattern)
            instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=rid)
            r, sat, arr = interp.execute(instr, state, wave_id=1)
            self.assertEqual(r, rid)
            self.assertEqual(state.get_vgpr(0), pattern)
            # Verify signaling NaN did not have its quiet bit set
            if pattern == self.F32_SNAN:
                self.assertEqual(state.get_vgpr(0) & 0x0040_0000, 0)

    def test_packed_f16_bit_patterns(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, self.F16_PACKED_ONES)
        interp.execute(DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=20), state, wave_id=1)
        self.assertEqual(state.get_vgpr(0), self.F16_PACKED_ONES)

        state.set_vgpr(0, self.F16_PACKED_NANS)
        interp.execute(DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=21), state, wave_id=1)
        self.assertEqual(state.get_vgpr(0), self.F16_PACKED_NANS)

    def test_simd_nan_transmission_from_first_active_lane(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x4)  # Lane 2 active
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_lane_vgpr(lane, 0, 0)
        state.set_lane_vgpr(2, 0, self.F32_CUSTOM_NAN_PAYLOAD)

        instr = DSGWSBarrierInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=30)
        res = interp.execute(instr, state, wave_id=2)

        self.assertIsNotNone(res)
        first_lane, rid, sat, arr = res  # type: ignore
        self.assertEqual(first_lane, 2)
        self.assertEqual(rid, 30)
        self.assertEqual(state.get_lane_vgpr(2, 0), self.F32_CUSTOM_NAN_PAYLOAD)


if __name__ == "__main__":
    unittest.main()
