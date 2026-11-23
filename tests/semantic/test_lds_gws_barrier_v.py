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




# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) GWS Barrier V opcode."""
    DS_GWS_BARRIER_V = "ds_gws_barrier_v"


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
class DSGWSBarrierVInstruction:
    """Symbolic AST node for ds_gws_barrier_v instruction."""
    vdata0: Optional[RegisterOperand] = None
    offset: int = 0
    gds: bool = True
    opcode: DSOpcode = DSOpcode.DS_GWS_BARRIER_V

    def __post_init__(self) -> None:
        if self.vdata0 is not None and self.vdata0.reg_type != RegType.VGPR:
            raise ValueError(f"vdata0 must be a VGPR, got {self.vdata0.reg_type}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(
                f"offset out of bounds: {self.offset} (expected 0 <= offset <= {OFFSET16_MASK})"
            )
        if not self.gds:
            raise ValueError("gds modifier is mandatory for GWS instructions")

    def disassemble(self) -> str:
        """Emit canonical AMD assembly string."""
        parts = [self.opcode.value]
        if self.vdata0 is not None:
            parts.append(str(self.vdata0))
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return f"{parts[0]} {', '.join(parts[1:])}"


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        instr = DSGWSBarrierVInstruction(offset=0, gds=True)
        self.assertEqual(instr.opcode, DSOpcode.DS_GWS_BARRIER_V)
        self.assertIsNone(instr.vdata0)
        self.assertEqual(instr.offset, 0)
        self.assertTrue(instr.gds)

    def test_valid_ast_with_vdata_and_offset(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        instr = DSGWSBarrierVInstruction(vdata0=v0, offset=8, gds=True)
        self.assertEqual(instr.vdata0.index, 0)
        self.assertEqual(instr.offset, 8)
        self.assertTrue(instr.gds)

    def test_disassembly_formatting(self) -> None:
        instr = DSGWSBarrierVInstruction(offset=0, gds=True)
        self.assertEqual(instr.disassemble(), "ds_gws_barrier_v gds")

        v1 = RegisterOperand(1, RegType.VGPR)
        instr_off = DSGWSBarrierVInstruction(vdata0=v1, offset=16, gds=True)
        self.assertEqual(instr_off.disassemble(), "ds_gws_barrier_v v[1], offset:16, gds")

    def test_rejection_sgpr_vdata0(self) -> None:
        s0 = RegisterOperand(0, RegType.SGPR)
        with self.assertRaises(ValueError):
            DSGWSBarrierVInstruction(vdata0=s0, offset=0, gds=True)

    def test_rejection_out_of_bounds_offset(self) -> None:
        with self.assertRaises(ValueError):
            DSGWSBarrierVInstruction(offset=-1, gds=True)
        with self.assertRaises(ValueError):
            DSGWSBarrierVInstruction(offset=65536, gds=True)

    def test_rejection_missing_gds_modifier(self) -> None:
        with self.assertRaises(ValueError):
            DSGWSBarrierVInstruction(offset=0, gds=False)

    def test_register_operand_validation(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)




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
    """Deterministic scalar instruction interpreter for ds_gws_barrier_v."""

    def execute(
        self,
        instr: DSGWSBarrierVInstruction,
        state: ScalarState,
    ) -> Tuple[int, int, list[int]]:
        """Execute a ds_gws_barrier_v instruction on scalar state.

        Returns:
            Tuple[int, int, list[int]]: (resource_id, total_releases, woken_waves)
        """
        m0 = state.get_m0()
        rid, releases, woken = state.gws.barrier_v(
            offset=instr.offset,
            m0=m0,
        )
        return rid, releases, woken


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter()

    def test_scalar_barrier_v_execution(self) -> None:
        self.state.gws.barriers[3].arrive_and_wait(1)
        self.state.gws.barriers[3].arrive_and_wait(2)

        instr = DSGWSBarrierVInstruction(offset=3)
        rid, rels, woken = self.interp.execute(instr, self.state)

        self.assertEqual(rid, 3)
        self.assertEqual(rels, 1)
        self.assertEqual(woken, [1, 2])
        self.assertEqual(self.state.gws.read_arrived(3), 0)
        self.assertEqual(self.state.gws.read_releases(3), 1)

    def test_scalar_with_offset_and_m0(self) -> None:
        self.state.set_m0(12)
        instr = DSGWSBarrierVInstruction(offset=4)
        rid, rels, woken = self.interp.execute(instr, self.state)
        self.assertEqual(rid, 16)
        self.assertEqual(rels, 1)
        self.assertEqual(self.state.get_m0(), 12)

    def test_scalar_registers_unmodified(self) -> None:
        for i in range(DEFAULT_NUM_VGPRS):
            self.state.set_vgpr(i, 0x7700 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.state.set_sgpr(i, 0x8800 + i)
        self.state.set_m0(7)

        instr = DSGWSBarrierVInstruction(offset=2)
        self.interp.execute(instr, self.state)

        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(self.state.get_vgpr(i), 0x7700 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(self.state.get_sgpr(i), 0x8800 + i)
        self.assertEqual(self.state.get_m0(), 7)

    def test_scalar_multiple_releases(self) -> None:
        instr = DSGWSBarrierVInstruction(offset=0)
        for i in range(1, 4):
            rid, rels, _ = self.interp.execute(instr, self.state)
            self.assertEqual(rid, 0)
            self.assertEqual(rels, i)
        self.assertEqual(self.state.gws.read_releases(0), 3)




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
    """SIMD multi-lane instruction interpreter for ds_gws_barrier_v."""

    def execute(
        self,
        instr: DSGWSBarrierVInstruction,
        state: SIMDState,
    ) -> Optional[Tuple[int, int, int, list[int]]]:
        """Execute ds_gws_barrier_v across SIMD lanes with EXEC masking.

        In RDNA2 hardware, ds_gws_barrier_v executes using the first active lane:
        first_lane = find_first_valid(EXEC mask).

        Returns:
            Optional[Tuple[int, int, int, list[int]]]: (first_lane, resource_id, total_releases, woken_waves)
            or None if the wave is inactive (EXEC == 0).
        """
        first_lane = state.find_first_active_lane()
        if first_lane is None:
            return None

        m0 = state.get_m0()
        rid, releases, woken = state.gws.barrier_v(
            offset=instr.offset,
            m0=m0,
        )
        return first_lane, rid, releases, woken


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        instr = DSGWSBarrierVInstruction(offset=0)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, rels, woken = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 0)
        self.assertEqual(rels, 1)

    def test_wave64_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        interp = SIMDInterpreter()

        instr = DSGWSBarrierVInstruction(offset=10)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, rels, woken = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 10)
        self.assertEqual(rels, 1)

    def test_wave32_divergent_mask_skips_inactive_lanes(self) -> None:
        mask = 0xFFFFFFF8  # Lanes 0..2 inactive, lane 3 active
        state = SIMDState(num_lanes=32, exec_mask=mask)
        interp = SIMDInterpreter()

        instr = DSGWSBarrierVInstruction(offset=5)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, rels, woken = res  # type: ignore
        self.assertEqual(first_lane, 3)
        self.assertEqual(rid, 5)

    def test_wave64_divergent_mask_high_lanes_only(self) -> None:
        mask = 1 << 50
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        instr = DSGWSBarrierVInstruction(offset=25)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, rels, woken = res  # type: ignore
        self.assertEqual(first_lane, 50)
        self.assertEqual(rid, 25)

    def test_zero_exec_mask_no_operation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        instr = DSGWSBarrierVInstruction(offset=2)
        res = interp.execute(instr, state)

        self.assertIsNone(res)
        self.assertEqual(state.gws.read_releases(2), 0)

    def test_registers_preserved_across_all_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            for r in range(10):
                state.set_lane_vgpr(lane, r, (lane << 16) | r)

        instr = DSGWSBarrierVInstruction(offset=4)
        interp.execute(instr, state)

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

        v0 = RegisterOperand(0, RegType.VGPR)
        instr_v0 = DSGWSBarrierVInstruction(vdata0=v0, offset=1)
        interp.execute(instr_v0, state)
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        self.assertEqual(state.get_m0(), 5)

        v255 = RegisterOperand(255, RegType.VGPR)
        instr_v255 = DSGWSBarrierVInstruction(vdata0=v255, offset=2)
        interp.execute(instr_v255, state)
        self.assertEqual(state.get_vgpr(255), 0x87654321)
        self.assertEqual(state.get_m0(), 5)

    def test_scalar_full_register_file_undisturbed(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        for i in range(DEFAULT_NUM_VGPRS):
            state.set_vgpr(i, 0xC0000000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            state.set_sgpr(i, 0xD0000000 | i)
        state.set_m0(0x19)

        instr = DSGWSBarrierVInstruction(offset=8)
        interp.execute(instr, state)

        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(state.get_vgpr(i), 0xC0000000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(state.get_sgpr(i), 0xD0000000 | i)
        self.assertEqual(state.get_m0(), 0x19)

    def test_scalar_repeated_operand_use_across_barriers(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        common_val = 0xFEEDFACE
        state.set_vgpr(5, common_val)
        v5 = RegisterOperand(5, RegType.VGPR)

        for i in range(6):
            instr = DSGWSBarrierVInstruction(vdata0=v5, offset=i)
            interp.execute(instr, state)
            self.assertEqual(state.get_vgpr(5), common_val)

        for i in range(6):
            self.assertEqual(state.gws.read_releases(i), 1)

    def test_simd_multi_lane_preservation_under_divergence(self) -> None:
        mask = 0x3333333333333333
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            for r in range(16):
                state.set_lane_vgpr(lane, r, ((lane + 1) * 200) + r)

        instr = DSGWSBarrierVInstruction(offset=14)
        res = interp.execute(instr, state)
        self.assertIsNotNone(res)

        for lane in range(64):
            for r in range(16):
                expected = ((lane + 1) * 200) + r
                self.assertEqual(state.get_lane_vgpr(lane, r), expected)


if __name__ == "__main__":
    unittest.main()
