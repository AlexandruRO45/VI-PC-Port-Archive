from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_gws_init.

RDNA2 / GFX10 Data Share Global Wave Sync (GWS) Initialization:
Initializes a dedicated Global Wave Sync (GWS) resource counter / barrier
located in Global Data Share (GDS) hardware.

Syntax:
    ds_gws_init vdata0 [offset:offset16] gds

Semantics:
    resource_id = (m0 + (offset16 & 0xFFFF)) & 0x3F
    first_active_lane = ctz(EXEC)
    if EXEC != 0:
        val = VGPR[first_active_lane][vdata0]
        GWS[resource_id] = val & 0xFFFFFFFF
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 GWS Architectural Constants
GWS_NUM_RESOURCES: int = 64         # 64 hardware GWS resource slots on GFX10
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


def evaluate_gws_init_status(
    value: int,
    offset: int = 0,
    m0: int = 0,
    exec_mask: int = 0xFFFFFFFF,
) -> dict[str, Union[bool, int]]:
    """Architectural evaluation of status flags and condition codes for ds_gws_init."""
    val = u32(value)
    eff_16 = (m0 & OFFSET16_MASK) + (offset & OFFSET16_MASK)
    eff_32 = m0 + (offset & OFFSET16_MASK)

    return {
        "zero_flag": (val == 0),
        "sign_flag": bool(val & 0x8000_0000),
        "offset_carry_out_16": (eff_16 > OFFSET16_MASK),
        "offset_carry_out_32": (eff_32 > MASK32),
        "active_wave": (exec_mask != 0),
        "vcc_unmodified": True,
        "scc_unmodified": True,
    }


def ref_ds_gws_init(init_val: int) -> int:
    """Pure mathematical reference oracle for ds_gws_init.

    Initializes the GWS resource state with the 32-bit unsigned value.
    Returns the normalized 32-bit initial value.
    """
    return u32(init_val)


@dataclass
class GWSResource:
    """Architectural state representation of an individual GWS resource."""
    counter: int = 0
    initialized: bool = False
    init_count: int = 0

    def init(self, val: int) -> int:
        """Initialize the resource counter."""
        self.counter = u32(val)
        self.initialized = True
        self.init_count += 1
        return self.counter

    def reset(self) -> None:
        """Reset resource state to defaults."""
        self.counter = 0
        self.initialized = False
        self.init_count = 0


class GWSMemory:
    """Architectural memory model representing the Global Wave Sync (GWS) subsystem."""

    def __init__(self, num_resources: int = GWS_NUM_RESOURCES) -> None:
        if num_resources <= 0 or (num_resources & (num_resources - 1)) != 0:
            raise ValueError(f"num_resources must be a positive power of two, got {num_resources}")
        self.num_resources = num_resources
        self.mask = num_resources - 1
        self.resources: list[GWSResource] = [GWSResource() for _ in range(num_resources)]

    def read_resource(self, resource_id: int) -> int:
        """Read the 32-bit counter from a GWS resource slot."""
        idx = resource_id & self.mask
        return self.resources[idx].counter

    def write_resource(self, resource_id: int, val: int) -> None:
        """Directly write a value to a GWS resource slot."""
        idx = resource_id & self.mask
        self.resources[idx].counter = u32(val)

    def init_resource(self, offset: int = 0, m0: int = 0, value: int = 0) -> Tuple[int, int]:
        """Execute ds_gws_init on the targeted GWS resource.

        Returns:
            Tuple[int, int]: (resource_id, initialized_value)
        """
        rid = calc_gws_resource_id(offset=offset, m0=m0)
        idx = rid & self.mask
        val = self.resources[idx].init(value)
        return rid, val


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) GWS opcodes."""
    DS_GWS_INIT = "ds_gws_init"


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
class DSGWSInitInstruction:
    """Symbolic AST node for ds_gws_init instruction."""
    vdata0: RegisterOperand
    offset: int = 0
    gds: bool = True
    opcode: DSOpcode = DSOpcode.DS_GWS_INIT

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
    """Deterministic scalar instruction interpreter for ds_gws_init."""

    def execute(self, instr: DSGWSInitInstruction, state: ScalarState) -> Tuple[int, int]:
        """Execute a single ds_gws_init instruction on scalar state.

        Returns:
            Tuple[int, int]: (resource_id, initialized_value)
        """
        # 1. Latch inputs prior to any execution
        vdata0 = state.get_reg(instr.vdata0)
        m0 = state.get_m0()

        # 2. Perform GWS resource initialization
        rid, val = state.gws.init_resource(
            offset=instr.offset,
            m0=m0,
            value=vdata0,
        )

        # ds_gws_init produces no register writeback; registers are preserved
        return rid, val


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
        # Lowest set bit index
        return (self.exec_mask & -self.exec_mask).bit_length() - 1


class SIMDInterpreter:
    """SIMD multi-lane instruction interpreter for ds_gws_init."""

    def execute(
        self,
        instr: DSGWSInitInstruction,
        state: SIMDState,
    ) -> Optional[Tuple[int, int, int]]:
        """Execute ds_gws_init across SIMD lanes with EXEC masking.

        In RDNA2 hardware, ds_gws_init operates on the first active thread in the wavefront:
        first_lane = find_first_valid(EXEC mask).
        The initialization data is pulled from that thread:
        value = DATA[thread: first_lane].

        Returns:
            Optional[Tuple[int, int, int]]: (first_lane, resource_id, initialized_value),
            or None if the wave is inactive (EXEC == 0).
        """
        first_lane = state.find_first_active_lane()
        if first_lane is None:
            return None

        # Latch vdata0 from the first active lane and m0 from scalar state
        vdata0 = state.get_lane_vgpr(first_lane, instr.vdata0.index)
        m0 = state.get_m0()

        # Initialize the GWS resource
        rid, val = state.gws.init_resource(
            offset=instr.offset,
            m0=m0,
            value=vdata0,
        )

        return first_lane, rid, val


class TestGWSInitReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_basic_values(self) -> None:
        self.assertEqual(ref_ds_gws_init(0), 0)
        self.assertEqual(ref_ds_gws_init(1), 1)
        self.assertEqual(ref_ds_gws_init(100), 100)
        self.assertEqual(ref_ds_gws_init(0x12345678), 0x12345678)
        self.assertEqual(ref_ds_gws_init(0xFFFFFFFF), 0xFFFFFFFF)

    def test_oracle_32bit_normalization(self) -> None:
        self.assertEqual(ref_ds_gws_init(0x1_0000_0000), 0)
        self.assertEqual(ref_ds_gws_init(0x1_ABCD_EF01), 0xABCD_EF01)
        self.assertEqual(ref_ds_gws_init(-1), 0xFFFFFFFF)
        self.assertEqual(ref_ds_gws_init(-100), 0xFFFFFF9C)

    def test_gws_resource_id_calculation(self) -> None:
        # Default offset=0, m0=0 -> resource 0
        self.assertEqual(calc_gws_resource_id(0, 0), 0)

        # Offset within 0..63
        self.assertEqual(calc_gws_resource_id(5, 0), 5)
        self.assertEqual(calc_gws_resource_id(63, 0), 63)

        # Modulo 64 wrapping
        self.assertEqual(calc_gws_resource_id(64, 0), 0)
        self.assertEqual(calc_gws_resource_id(65, 0), 1)
        self.assertEqual(calc_gws_resource_id(127, 0), 63)
        self.assertEqual(calc_gws_resource_id(128, 0), 0)

        # m0 offset combined with immediate offset
        self.assertEqual(calc_gws_resource_id(10, 4), 14)
        self.assertEqual(calc_gws_resource_id(60, 5), 1)  # (60 + 5) = 65 -> 1

        # 16-bit offset masking
        self.assertEqual(calc_gws_resource_id(0x10005, 0), 5)

    def test_effective_offset_calculation(self) -> None:
        self.assertEqual(calc_gws_effective_offset(0x100, 0x20), 0x120)
        self.assertEqual(calc_gws_effective_offset(0xFFFF, 1), 0x10000)

    def test_gws_memory_model_init_and_read(self) -> None:
        gws = GWSMemory()
        self.assertEqual(gws.num_resources, 64)

        # Initially, all resources have counter 0 and are uninitialized
        for r in range(64):
            self.assertEqual(gws.read_resource(r), 0)
            self.assertFalse(gws.resources[r].initialized)

        # Init resource 7 with value 0x42
        rid, val = gws.init_resource(offset=7, m0=0, value=0x42)
        self.assertEqual(rid, 7)
        self.assertEqual(val, 0x42)
        self.assertEqual(gws.read_resource(7), 0x42)
        self.assertTrue(gws.resources[7].initialized)
        self.assertEqual(gws.resources[7].init_count, 1)

        # Other resources unchanged
        self.assertEqual(gws.read_resource(6), 0)
        self.assertEqual(gws.read_resource(8), 0)

    def test_gws_memory_reinitialization(self) -> None:
        gws = GWSMemory()
        rid, val1 = gws.init_resource(offset=15, m0=0, value=100)
        self.assertEqual(val1, 100)
        self.assertEqual(gws.resources[15].init_count, 1)

        rid, val2 = gws.init_resource(offset=15, m0=0, value=200)
        self.assertEqual(val2, 200)
        self.assertEqual(gws.read_resource(15), 200)
        self.assertEqual(gws.resources[15].init_count, 2)

    def test_gws_memory_reset(self) -> None:
        gws = GWSMemory()
        gws.init_resource(offset=3, value=999)
        self.assertEqual(gws.read_resource(3), 999)
        gws.resources[3].reset()
        self.assertEqual(gws.read_resource(3), 0)
        self.assertFalse(gws.resources[3].initialized)
        self.assertEqual(gws.resources[3].init_count, 0)


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        instr = DSGWSInitInstruction(vdata0=v0, offset=0, gds=True)

        self.assertEqual(instr.opcode, DSOpcode.DS_GWS_INIT)
        self.assertEqual(instr.vdata0.index, 0)
        self.assertEqual(instr.offset, 0)
        self.assertTrue(instr.gds)

    def test_valid_ast_with_offset(self) -> None:
        v5 = RegisterOperand(5, RegType.VGPR)
        instr = DSGWSInitInstruction(vdata0=v5, offset=16, gds=True)

        self.assertEqual(instr.vdata0.index, 5)
        self.assertEqual(instr.offset, 16)
        self.assertTrue(instr.gds)

    def test_disassembly_formatting(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v7 = RegisterOperand(7, RegType.VGPR)

        instr_no_offset = DSGWSInitInstruction(vdata0=v0, offset=0, gds=True)
        self.assertEqual(instr_no_offset.disassemble(), "ds_gws_init v[0], gds")

        instr_with_offset = DSGWSInitInstruction(vdata0=v7, offset=32, gds=True)
        self.assertEqual(instr_with_offset.disassemble(), "ds_gws_init v[7], offset:32, gds")

    def test_rejection_sgpr_vdata0(self) -> None:
        s0 = RegisterOperand(0, RegType.SGPR)
        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=s0, offset=0, gds=True)

    def test_rejection_out_of_bounds_offset(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=v0, offset=-1, gds=True)

        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=v0, offset=65536, gds=True)

    def test_rejection_missing_gds_modifier(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=v0, offset=0, gds=False)

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

    def test_scalar_gws_init_basic(self) -> None:
        # v3 holds initial value 100
        self.state.set_vgpr(3, 100)
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(3, RegType.VGPR), offset=0)

        rid, val = self.interp.execute(instr, self.state)
        self.assertEqual(rid, 0)
        self.assertEqual(val, 100)
        self.assertEqual(self.state.gws.read_resource(0), 100)
        self.assertTrue(self.state.gws.resources[0].initialized)

        # Registers must remain unmodified
        self.assertEqual(self.state.get_vgpr(3), 100)

    def test_scalar_gws_init_with_offset(self) -> None:
        self.state.set_vgpr(1, 0x1234)
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=12)

        rid, val = self.interp.execute(instr, self.state)
        self.assertEqual(rid, 12)
        self.assertEqual(val, 0x1234)
        self.assertEqual(self.state.gws.read_resource(12), 0x1234)

    def test_scalar_gws_init_with_m0(self) -> None:
        self.state.set_m0(20)
        self.state.set_vgpr(2, 555)
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=5)

        rid, val = self.interp.execute(instr, self.state)
        self.assertEqual(rid, 25)
        self.assertEqual(val, 555)
        self.assertEqual(self.state.gws.read_resource(25), 555)
        self.assertEqual(self.state.get_m0(), 20)

    def test_scalar_sequential_inits(self) -> None:
        # Initialize resources 0, 1, 2 sequentially
        for i in range(3):
            self.state.set_vgpr(i, 1000 + i)
            instr = DSGWSInitInstruction(vdata0=RegisterOperand(i, RegType.VGPR), offset=i)
            rid, val = self.interp.execute(instr, self.state)
            self.assertEqual(rid, i)
            self.assertEqual(val, 1000 + i)

        for i in range(3):
            self.assertEqual(self.state.gws.read_resource(i), 1000 + i)

    def test_scalar_registers_unmodified(self) -> None:
        # Set all registers to known sentinels
        for i in range(DEFAULT_NUM_VGPRS):
            self.state.set_vgpr(i, 0x1000 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.state.set_sgpr(i, 0x2000 + i)
        self.state.set_m0(7)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(42, RegType.VGPR), offset=3)
        self.interp.execute(instr, self.state)

        # Check all registers are untouched
        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(self.state.get_vgpr(i), 0x1000 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(self.state.get_sgpr(i), 0x2000 + i)
        self.assertEqual(self.state.get_m0(), 7)

    def test_scalar_large_values_and_wrapping(self) -> None:
        self.state.set_vgpr(0, 0xFFFFFFFF)
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=70)  # 70 % 64 = 6
        rid, val = self.interp.execute(instr, self.state)
        self.assertEqual(rid, 6)
        self.assertEqual(val, 0xFFFFFFFF)
        self.assertEqual(self.state.gws.read_resource(6), 0xFFFFFFFF)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        # Each lane has distinct vdata0 in v2: lane 0 has 1000, lane 1 has 1001, etc.
        for lane in range(32):
            state.set_lane_vgpr(lane, 2, 1000 + lane)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=0)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, val = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 0)
        self.assertEqual(val, 1000)  # Lane 0's value must be selected
        self.assertEqual(state.gws.read_resource(0), 1000)

    def test_wave64_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        interp = SIMDInterpreter()

        for lane in range(64):
            state.set_lane_vgpr(lane, 4, 5000 + lane)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(4, RegType.VGPR), offset=10)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, val = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 10)
        self.assertEqual(val, 5000)
        self.assertEqual(state.gws.read_resource(10), 5000)

    def test_wave32_divergent_mask_skips_inactive_lanes(self) -> None:
        # Lanes 0..3 inactive, lane 4 is first active (mask: 0xFFFFFFF0)
        mask = 0xFFFFFFF0
        state = SIMDState(num_lanes=32, exec_mask=mask)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_lane_vgpr(lane, 1, 100 * (lane + 1))

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=5)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, val = res  # type: ignore
        self.assertEqual(first_lane, 4)
        self.assertEqual(rid, 5)
        self.assertEqual(val, 500)  # Lane 4's value (100 * 5)
        self.assertEqual(state.gws.read_resource(5), 500)

    def test_wave64_divergent_mask_high_lanes_only(self) -> None:
        # Only lane 48 active
        mask = 1 << 48
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(48, 7, 0xCAFEBABE)
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(7, RegType.VGPR), offset=33)

        res = interp.execute(instr, state)
        self.assertIsNotNone(res)
        first_lane, rid, val = res  # type: ignore
        self.assertEqual(first_lane, 48)
        self.assertEqual(rid, 33)
        self.assertEqual(val, 0xCAFEBABE)
        self.assertEqual(state.gws.read_resource(33), 0xCAFEBABE)

    def test_zero_exec_mask_no_operation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 1, 9999)
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=1)

        res = interp.execute(instr, state)
        self.assertIsNone(res)
        # GWS resource 1 must remain untouched and uninitialized
        self.assertEqual(state.gws.read_resource(1), 0)
        self.assertFalse(state.gws.resources[1].initialized)
        self.assertEqual(state.gws.resources[1].init_count, 0)

    def test_registers_preserved_across_all_lanes(self) -> None:
        # Alternating mask (even lanes active)
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask)
        interp = SIMDInterpreter()

        # Seed VGPRs across all lanes
        for lane in range(64):
            for reg in range(10):
                state.set_lane_vgpr(lane, reg, (lane << 16) | reg)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=2)
        interp.execute(instr, state)

        # Verify no VGPR in any lane was altered
        for lane in range(64):
            for reg in range(10):
                expected = (lane << 16) | reg
                self.assertEqual(state.get_lane_vgpr(lane, reg), expected)


class TestRegisterAliasingAndOperandPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_scalar_operand_preservation_boundary_registers(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        # Test first VGPR (v0) and last VGPR (v255)
        state.set_vgpr(0, 0xAA55AA55)
        state.set_vgpr(255, 0x12345678)
        state.set_m0(4)

        instr_v0 = DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=1)
        rid0, val0 = interp.execute(instr_v0, state)
        self.assertEqual(rid0, 5)
        self.assertEqual(val0, 0xAA55AA55)
        self.assertEqual(state.get_vgpr(0), 0xAA55AA55)
        self.assertEqual(state.get_m0(), 4)

        instr_v255 = DSGWSInitInstruction(vdata0=RegisterOperand(255, RegType.VGPR), offset=2)
        rid255, val255 = interp.execute(instr_v255, state)
        self.assertEqual(rid255, 6)
        self.assertEqual(val255, 0x12345678)
        self.assertEqual(state.get_vgpr(255), 0x12345678)
        self.assertEqual(state.get_m0(), 4)

    def test_scalar_full_register_file_undisturbed(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        # Seed every register with a unique signature
        for i in range(DEFAULT_NUM_VGPRS):
            state.set_vgpr(i, 0xDEAD0000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            state.set_sgpr(i, 0xBEEF0000 | i)
        state.set_m0(0x42)

        target_reg = 127
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(target_reg, RegType.VGPR), offset=15)
        rid, val = interp.execute(instr, state)

        self.assertEqual(rid, (0x42 + 15) & 0x3F)
        self.assertEqual(val, 0xDEAD0000 | target_reg)

        # Confirm all 256 VGPRs and 106 SGPRs remain perfectly intact
        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(state.get_vgpr(i), 0xDEAD0000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(state.get_sgpr(i), 0xBEEF0000 | i)
        self.assertEqual(state.get_m0(), 0x42)

    def test_scalar_repeated_operand_use_across_inits(self) -> None:
        # A single VGPR used to initialize multiple GWS resources
        state = ScalarState()
        interp = ScalarInterpreter()

        common_val = 0xCAFE1234
        state.set_vgpr(10, common_val)

        for off in range(8):
            instr = DSGWSInitInstruction(vdata0=RegisterOperand(10, RegType.VGPR), offset=off)
            rid, val = interp.execute(instr, state)
            self.assertEqual(rid, off)
            self.assertEqual(val, common_val)
            self.assertEqual(state.get_vgpr(10), common_val)

        for off in range(8):
            self.assertEqual(state.gws.read_resource(off), common_val)

    def test_simd_multi_lane_preservation_under_divergence(self) -> None:
        # Wave64 with divergent EXEC mask (scattered lanes)
        scattered_mask = 0x9090909012345678
        state = SIMDState(num_lanes=64, exec_mask=scattered_mask)
        interp = SIMDInterpreter()

        # Initialize all registers in all lanes
        for lane in range(64):
            for r in range(16):
                state.set_lane_vgpr(lane, r, ((lane + 1) * 1000) + r)

        vdata_reg = 5
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(vdata_reg, RegType.VGPR), offset=20)
        res = interp.execute(instr, state)
        self.assertIsNotNone(res)

        # Check every lane's register file
        for lane in range(64):
            for r in range(16):
                expected = ((lane + 1) * 1000) + r
                self.assertEqual(state.get_lane_vgpr(lane, r), expected)


class TestConditionCodesStatusFlagsAndCarryBorrow(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_preserved_scalar(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        # Pre-seed condition codes with sentinel values
        state.vcc = 0xDEADBEEF_CAFEBABE
        state.scc = 1
        state.set_vgpr(0, 42)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=0)
        interp.execute(instr, state)

        # Neither VCC nor SCC should be modified by ds_gws_init
        self.assertEqual(state.vcc, 0xDEADBEEF_CAFEBABE)
        self.assertEqual(state.scc, 1)

    def test_vcc_and_scc_preserved_simd(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        state.vcc = 0x12345678
        state.scc = 0
        state.set_lane_vgpr(0, 1, 99)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=4)
        interp.execute(instr, state)

        self.assertEqual(state.vcc, 0x12345678)
        self.assertEqual(state.scc, 0)

    def test_no_carry_borrow_generated(self) -> None:
        # Carry and borrow generation is an arithmetic operation (VOP2 / SALU).
        # DS instructions never emit carry or borrow bits to condition code registers.
        state = ScalarState()
        interp = ScalarInterpreter()

        # Initialize with max values that in normal addition would produce carry
        state.set_vgpr(0, 0xFFFFFFFF)
        state.vcc = 0
        state.scc = 0

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=63)
        interp.execute(instr, state)

        self.assertEqual(state.vcc, 0)
        self.assertEqual(state.scc, 0)

    def test_status_flags_zero_and_sign(self) -> None:
        # Value 0: zero_flag=True, sign_flag=False
        st0 = evaluate_gws_init_status(value=0, offset=0, m0=0)
        self.assertTrue(st0["zero_flag"])
        self.assertFalse(st0["sign_flag"])

        # Value 100: zero_flag=False, sign_flag=False
        st1 = evaluate_gws_init_status(value=100, offset=0, m0=0)
        self.assertFalse(st1["zero_flag"])
        self.assertFalse(st1["sign_flag"])

        # Value 0x80000000 (sign bit set): zero_flag=False, sign_flag=True
        st_sign = evaluate_gws_init_status(value=0x80000000, offset=0, m0=0)
        self.assertFalse(st_sign["zero_flag"])
        self.assertTrue(st_sign["sign_flag"])

        # Value 0xFFFFFFFF: zero_flag=False, sign_flag=True
        st_max = evaluate_gws_init_status(value=0xFFFFFFFF, offset=0, m0=0)
        self.assertFalse(st_max["zero_flag"])
        self.assertTrue(st_max["sign_flag"])

    def test_status_flags_offset_carry_out(self) -> None:
        # No carry out
        st_no_carry = evaluate_gws_init_status(value=1, offset=10, m0=20)
        self.assertFalse(st_no_carry["offset_carry_out_16"])
        self.assertFalse(st_no_carry["offset_carry_out_32"])

        # 16-bit offset carry out (e.g. m0=0xFFF0 + offset=0x0020 = 0x10010 > 0xFFFF)
        st_carry_16 = evaluate_gws_init_status(value=1, offset=0x0020, m0=0xFFF0)
        self.assertTrue(st_carry_16["offset_carry_out_16"])
        self.assertFalse(st_carry_16["offset_carry_out_32"])

        # 32-bit wrap carry out
        st_carry_32 = evaluate_gws_init_status(value=1, offset=1, m0=0xFFFFFFFF)
        self.assertTrue(st_carry_32["offset_carry_out_32"])

    def test_status_flags_wave_activity(self) -> None:
        st_active = evaluate_gws_init_status(value=1, exec_mask=0x1)
        self.assertTrue(st_active["active_wave"])

        st_inactive = evaluate_gws_init_status(value=1, exec_mask=0x0)
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

    def test_signed_zeros_preservation(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        # +0.0 (0x00000000)
        state.set_vgpr(0, self.F32_POS_ZERO)
        interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=0), state)
        self.assertEqual(state.gws.read_resource(0), self.F32_POS_ZERO)

        # -0.0 (0x80000000) - sign bit must NOT be cleared
        state.set_vgpr(1, self.F32_NEG_ZERO)
        interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=1), state)
        self.assertEqual(state.gws.read_resource(1), self.F32_NEG_ZERO)

    def test_subnormals_preservation_no_ftz(self) -> None:
        # RDNA2 GWS data path is bit-exact; no Flush-To-Zero (FTZ) may occur
        state = ScalarState()
        interp = ScalarInterpreter()

        subnormals = [
            (2, self.F32_MIN_SUBNORMAL),
            (3, self.F32_MAX_SUBNORMAL),
            (4, self.F32_NEG_MIN_SUBNORMAL),
            (5, self.F32_NEG_MAX_SUBNORMAL),
        ]

        for rid, pattern in subnormals:
            state.set_vgpr(0, pattern)
            interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=rid), state)
            self.assertEqual(state.gws.read_resource(rid), pattern)
            self.assertNotEqual(state.gws.read_resource(rid), 0)

    def test_normal_boundaries_and_infinities(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        patterns = [
            (10, self.F32_MIN_NORMAL),
            (11, self.F32_ONE),
            (12, self.F32_NEG_ONE),
            (13, self.F32_MAX_NORMAL),
            (14, self.F32_NEG_MAX_NORMAL),
            (15, self.F32_POS_INF),
            (16, self.F32_NEG_INF),
        ]

        for rid, pattern in patterns:
            state.set_vgpr(0, pattern)
            interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=rid), state)
            self.assertEqual(state.gws.read_resource(rid), pattern)

    def test_nans_payload_preservation_no_quieting(self) -> None:
        # Signaling NaNs must NOT be quieted; payload bits must be preserved intact
        state = ScalarState()
        interp = ScalarInterpreter()

        nan_patterns = [
            (20, self.F32_QNAN_CANONICAL),
            (21, self.F32_SNAN),
            (22, self.F32_NEG_QNAN),
            (23, self.F32_CUSTOM_NAN_PAYLOAD),
        ]

        for rid, pattern in nan_patterns:
            state.set_vgpr(0, pattern)
            interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=rid), state)
            self.assertEqual(state.gws.read_resource(rid), pattern)
            # Verify signaling NaN did not have its quiet bit set
            if pattern == self.F32_SNAN:
                self.assertEqual(state.gws.read_resource(rid) & 0x0040_0000, 0)

    def test_packed_f16_bit_patterns(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, self.F16_PACKED_ONES)
        interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=30), state)
        self.assertEqual(state.gws.read_resource(30), self.F16_PACKED_ONES)

        state.set_vgpr(0, self.F16_PACKED_NANS)
        interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=31), state)
        self.assertEqual(state.gws.read_resource(31), self.F16_PACKED_NANS)

    def test_simd_nan_transmission_from_first_active_lane(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x8)  # Only lane 3 active
        interp = SIMDInterpreter()

        # Seed other lanes with 0, lane 3 with custom NaN
        for lane in range(32):
            state.set_lane_vgpr(lane, 0, 0)
        state.set_lane_vgpr(3, 0, self.F32_CUSTOM_NAN_PAYLOAD)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=40)
        res = interp.execute(instr, state)

        self.assertIsNotNone(res)
        first_lane, rid, val = res  # type: ignore
        self.assertEqual(first_lane, 3)
        self.assertEqual(rid, 40)
        self.assertEqual(val, self.F32_CUSTOM_NAN_PAYLOAD)
        self.assertEqual(state.gws.read_resource(40), self.F32_CUSTOM_NAN_PAYLOAD)


class TestNegativeControlsSyntaxRejectionAndMutations(unittest.TestCase):
    """Phase 8: Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_negative_and_out_of_bounds_registers(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)

        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)

        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.SGPR)

        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

    def test_syntax_rejection_sgpr_as_vdata0(self) -> None:
        s0 = RegisterOperand(0, RegType.SGPR)
        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=s0, offset=0, gds=True)

    def test_syntax_rejection_invalid_offset_bounds(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=v0, offset=-1, gds=True)

        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=v0, offset=0x10000, gds=True)

    def test_syntax_rejection_missing_mandatory_gds(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            DSGWSInitInstruction(vdata0=v0, offset=0, gds=False)

    def test_invalid_simd_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)

        with self.assertRaises(ValueError):
            SIMDState(num_lanes=48)

        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_invalid_gws_memory_size(self) -> None:
        with self.assertRaises(ValueError):
            GWSMemory(num_resources=0)

        with self.assertRaises(ValueError):
            GWSMemory(num_resources=-1)

        with self.assertRaises(ValueError):
            GWSMemory(num_resources=48)  # Not a power of two

    def test_mutation_detection_last_active_lane_elected(self) -> None:
        # Reference behavior: lane 0 elected.
        # Mutant: erroneously elects the highest active lane (lane 31).
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 5, 0x1111)
        state.set_lane_vgpr(31, 5, 0x9999)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(5, RegType.VGPR), offset=1)
        res = interp.execute(instr, state)
        self.assertIsNotNone(res)
        first_lane, _, val = res  # type: ignore

        # Reference must choose lane 0 and 0x1111
        self.assertEqual(first_lane, 0)
        self.assertEqual(val, 0x1111)

        # Mutant behavior check
        mutant_last_lane = 31
        mutant_val = state.get_lane_vgpr(mutant_last_lane, 5)
        self.assertNotEqual(mutant_last_lane, first_lane)
        self.assertNotEqual(mutant_val, val)

    def test_mutation_detection_multi_lane_overwriting(self) -> None:
        # A buggy interpreter that runs all active lanes sequentially in a loop
        # would overwrite the GWS resource with the last active lane's value.
        state = SIMDState(num_lanes=32, exec_mask=0b11)  # Lanes 0 and 1 active
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 1, 100)
        state.set_lane_vgpr(1, 1, 200)

        instr = DSGWSInitInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=7)
        interp.execute(instr, state)

        # Reference: resource 7 has lane 0's value (100)
        self.assertEqual(state.gws.read_resource(7), 100)

        # Simulated mutant loop:
        mutant_gws = GWSMemory()
        for lane in (0, 1):
            mutant_gws.init_resource(offset=7, value=state.get_lane_vgpr(lane, 1))

        self.assertNotEqual(mutant_gws.read_resource(7), state.gws.read_resource(7))
        self.assertEqual(mutant_gws.read_resource(7), 200)

    def test_mutation_detection_exec_zero_bypass(self) -> None:
        # Mutant ignores exec_mask == 0 and executes using lane 0
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        state.set_lane_vgpr(0, 2, 0xBEEF)
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(2, RegType.VGPR), offset=12)

        res = interp.execute(instr, state)
        self.assertIsNone(res)
        self.assertEqual(state.gws.read_resource(12), 0)

        # Mutant execution would yield 0xBEEF
        mutant_executed_val = 0xBEEF
        self.assertNotEqual(mutant_executed_val, state.gws.read_resource(12))

    def test_mutation_detection_dropped_offset(self) -> None:
        # Mutant ignores instruction offset and hardcodes resource 0
        state = ScalarState()
        interp = ScalarInterpreter()

        state.set_vgpr(1, 777)
        target_offset = 25
        instr = DSGWSInitInstruction(vdata0=RegisterOperand(1, RegType.VGPR), offset=target_offset)

        rid, val = interp.execute(instr, state)
        self.assertEqual(rid, target_offset)
        self.assertEqual(state.gws.read_resource(target_offset), 777)
        self.assertEqual(state.gws.read_resource(0), 0)

        # Mutant writing to 0 would fail
        mutant_rid = 0
        self.assertNotEqual(mutant_rid, rid)

    def test_mutation_detection_value_corruption(self) -> None:
        # Mutant inverts bits of value
        orig_val = 0x55AA55AA
        mutant_val = (~orig_val) & MASK32

        self.assertNotEqual(mutant_val, orig_val)
        ref_val = ref_ds_gws_init(orig_val)
        self.assertEqual(ref_val, orig_val)
        self.assertNotEqual(ref_val, mutant_val)

    def test_mutation_detection_snan_quieting(self) -> None:
        # Mutant sets the quiet bit on signaling NaNs
        snan = 0x7F800001
        mutant_quieted = snan | 0x00400000

        state = ScalarState()
        interp = ScalarInterpreter()
        state.set_vgpr(0, snan)

        interp.execute(DSGWSInitInstruction(vdata0=RegisterOperand(0, RegType.VGPR), offset=0), state)
        self.assertEqual(state.gws.read_resource(0), snan)
        self.assertNotEqual(state.gws.read_resource(0), mutant_quieted)


if __name__ == "__main__":
    unittest.main()
