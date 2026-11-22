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


if __name__ == "__main__":
    unittest.main()
