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


if __name__ == "__main__":
    unittest.main()
