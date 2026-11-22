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


def evaluate_gws_sema_status(
    counter: int,
    offset: int = 0,
    m0: int = 0,
    exec_mask: int = 0xFFFFFFFF,
) -> dict[str, Union[bool, int]]:
    """Architectural evaluation of status flags and condition codes for GWS semaphores."""
    cnt = u32(counter)
    eff_16 = (m0 & OFFSET16_MASK) + (offset & OFFSET16_MASK)
    eff_32 = m0 + (offset & OFFSET16_MASK)

    return {
        "semaphore_available": (cnt > 0),
        "counter_zero": (cnt == 0),
        "offset_carry_out_16": (eff_16 > OFFSET16_MASK),
        "offset_carry_out_32": (eff_32 > MASK32),
        "active_wave": (exec_mask != 0),
        "vcc_unmodified": True,
        "scc_unmodified": True,
    }


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


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) GWS Semaphore opcodes."""
    DS_GWS_SEMA_P = "ds_gws_sema_p"
    DS_GWS_SEMA_V = "ds_gws_sema_v"


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
class DSGWSSemaphoreInstruction:
    """Symbolic AST node for ds_gws_sema_p and ds_gws_sema_v instructions."""
    opcode: DSOpcode
    vdata0: Optional[RegisterOperand] = None
    offset: int = 0
    gds: bool = True

    def __post_init__(self) -> None:
        if self.opcode not in (DSOpcode.DS_GWS_SEMA_P, DSOpcode.DS_GWS_SEMA_V):
            raise ValueError(f"Invalid semaphore opcode: {self.opcode}")
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
    """Deterministic scalar instruction interpreter for ds_gws_sema_p and ds_gws_sema_v."""

    def execute(
        self,
        instr: DSGWSSemaphoreInstruction,
        state: ScalarState,
        wave_id: int = 0,
    ) -> Tuple[int, Union[bool, int], Union[int, Optional[int]]]:
        """Execute a semaphore instruction on scalar state.

        Returns:
            For ds_gws_sema_p: Tuple[int, bool, int] -> (resource_id, acquired, current_counter)
            For ds_gws_sema_v: Tuple[int, int, Optional[int]] -> (resource_id, new_counter, woken_wave_or_none)
        """
        m0 = state.get_m0()

        if instr.opcode == DSOpcode.DS_GWS_SEMA_P:
            rid, acquired, cnt = state.gws.sema_p(
                offset=instr.offset,
                m0=m0,
                wave_id=wave_id,
            )
            return rid, acquired, cnt
        else:
            rid, cnt, woken = state.gws.sema_v(
                offset=instr.offset,
                m0=m0,
            )
            return rid, cnt, woken


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
    """SIMD multi-lane instruction interpreter for ds_gws_sema_p and ds_gws_sema_v."""

    def execute(
        self,
        instr: DSGWSSemaphoreInstruction,
        state: SIMDState,
        wave_id: int = 0,
    ) -> Optional[Tuple[int, int, Union[bool, int], Union[int, Optional[int]]]]:
        """Execute a semaphore instruction across SIMD lanes with EXEC masking.

        In RDNA2 hardware, GWS semaphore instructions execute using the first active lane:
        first_lane = find_first_valid(EXEC mask).

        Returns:
            Optional[Tuple[int, int, ...]]: (first_lane, resource_id, outcome, state_detail)
            or None if the wave is inactive (EXEC == 0).
        """
        first_lane = state.find_first_active_lane()
        if first_lane is None:
            return None

        m0 = state.get_m0()

        if instr.opcode == DSOpcode.DS_GWS_SEMA_P:
            rid, acquired, cnt = state.gws.sema_p(
                offset=instr.offset,
                m0=m0,
                wave_id=wave_id,
            )
            return first_lane, rid, acquired, cnt
        else:
            rid, cnt, woken = state.gws.sema_v(
                offset=instr.offset,
                m0=m0,
            )
            return first_lane, rid, cnt, woken


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


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction_sema_p(self) -> None:
        instr = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=0, gds=True)
        self.assertEqual(instr.opcode, DSOpcode.DS_GWS_SEMA_P)
        self.assertIsNone(instr.vdata0)
        self.assertEqual(instr.offset, 0)
        self.assertTrue(instr.gds)

    def test_valid_ast_construction_sema_v_with_vdata(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        instr = DSGWSSemaphoreInstruction(
            opcode=DSOpcode.DS_GWS_SEMA_V,
            vdata0=v0,
            offset=12,
            gds=True,
        )
        self.assertEqual(instr.opcode, DSOpcode.DS_GWS_SEMA_V)
        self.assertEqual(instr.vdata0.index, 0)
        self.assertEqual(instr.offset, 12)
        self.assertTrue(instr.gds)

    def test_disassembly_formatting(self) -> None:
        instr_p = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=0, gds=True)
        self.assertEqual(instr_p.disassemble(), "ds_gws_sema_p gds")

        instr_p_off = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=4, gds=True)
        self.assertEqual(instr_p_off.disassemble(), "ds_gws_sema_p offset:4, gds")

        v1 = RegisterOperand(1, RegType.VGPR)
        instr_v_vdata = DSGWSSemaphoreInstruction(
            opcode=DSOpcode.DS_GWS_SEMA_V,
            vdata0=v1,
            offset=8,
            gds=True,
        )
        self.assertEqual(instr_v_vdata.disassemble(), "ds_gws_sema_v v[1], offset:8, gds")

    def test_rejection_sgpr_vdata0(self) -> None:
        s0 = RegisterOperand(0, RegType.SGPR)
        with self.assertRaises(ValueError):
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, vdata0=s0, offset=0, gds=True)

    def test_rejection_out_of_bounds_offset(self) -> None:
        with self.assertRaises(ValueError):
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=-1, gds=True)
        with self.assertRaises(ValueError):
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=65536, gds=True)

    def test_rejection_missing_gds_modifier(self) -> None:
        with self.assertRaises(ValueError):
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=0, gds=False)

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

    def test_scalar_sema_p_and_v_cycle(self) -> None:
        # Initially counter 0 on resource 5
        instr_p = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=5)
        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=5)

        # 1. P operation blocks because counter is 0
        rid_p, acq1, cnt1 = self.interp.execute(instr_p, self.state, wave_id=10)
        self.assertEqual(rid_p, 5)
        self.assertFalse(acq1)
        self.assertEqual(cnt1, 0)
        self.assertEqual(len(self.state.gws.semaphores[5].waiting_waves), 1)

        # 2. V operation wakes wave 10
        rid_v, cnt_v, woken = self.interp.execute(instr_v, self.state)
        self.assertEqual(rid_v, 5)
        self.assertEqual(woken, 10)
        self.assertEqual(len(self.state.gws.semaphores[5].waiting_waves), 0)

        # 3. Another V operation increments counter to 1
        rid_v2, cnt_v2, woken2 = self.interp.execute(instr_v, self.state)
        self.assertEqual(cnt_v2, 1)
        self.assertIsNone(woken2)
        self.assertEqual(self.state.gws.read_counter(5), 1)

        # 4. P operation succeeds immediately
        rid_p2, acq2, cnt2 = self.interp.execute(instr_p, self.state, wave_id=11)
        self.assertTrue(acq2)
        self.assertEqual(cnt2, 0)
        self.assertEqual(self.state.gws.read_counter(5), 0)

    def test_scalar_with_offset_and_m0(self) -> None:
        self.state.set_m0(16)
        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=4)
        rid, cnt, _ = self.interp.execute(instr_v, self.state)
        self.assertEqual(rid, 20)
        self.assertEqual(cnt, 1)
        self.assertEqual(self.state.get_m0(), 16)

    def test_scalar_registers_unmodified(self) -> None:
        for i in range(DEFAULT_NUM_VGPRS):
            self.state.set_vgpr(i, 0x5500 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.state.set_sgpr(i, 0x6600 + i)
        self.state.set_m0(3)

        instr = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=1)
        self.interp.execute(instr, self.state)

        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(self.state.get_vgpr(i), 0x5500 + i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(self.state.get_sgpr(i), 0x6600 + i)
        self.assertEqual(self.state.get_m0(), 3)

    def test_scalar_multi_resource_isolation(self) -> None:
        instr_v0 = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=0)
        instr_v1 = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=1)

        self.interp.execute(instr_v0, self.state)
        self.interp.execute(instr_v0, self.state)
        self.interp.execute(instr_v1, self.state)

        self.assertEqual(self.state.gws.read_counter(0), 2)
        self.assertEqual(self.state.gws.read_counter(1), 1)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=0)
        res = interp.execute(instr_v, state)

        self.assertIsNotNone(res)
        first_lane, rid, cnt, woken = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 0)
        self.assertEqual(cnt, 1)
        self.assertEqual(state.gws.read_counter(0), 1)

    def test_wave64_all_lanes_active_lane0_elected(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        interp = SIMDInterpreter()

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=7)
        res = interp.execute(instr_v, state)

        self.assertIsNotNone(res)
        first_lane, rid, cnt, woken = res  # type: ignore
        self.assertEqual(first_lane, 0)
        self.assertEqual(rid, 7)
        self.assertEqual(cnt, 1)

    def test_wave32_divergent_mask_skips_inactive_lanes(self) -> None:
        # Lanes 0..5 inactive, lane 6 active
        mask = 0xFFFFFFC0
        state = SIMDState(num_lanes=32, exec_mask=mask)
        interp = SIMDInterpreter()

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=11)
        res = interp.execute(instr_v, state)

        self.assertIsNotNone(res)
        first_lane, rid, cnt, woken = res  # type: ignore
        self.assertEqual(first_lane, 6)
        self.assertEqual(rid, 11)

    def test_wave64_divergent_mask_high_lanes_only(self) -> None:
        mask = 1 << 45
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=22)
        res = interp.execute(instr_v, state)

        self.assertIsNotNone(res)
        first_lane, rid, cnt, woken = res  # type: ignore
        self.assertEqual(first_lane, 45)
        self.assertEqual(rid, 22)

    def test_zero_exec_mask_no_operation(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=3)
        res = interp.execute(instr_v, state)

        self.assertIsNone(res)
        # Counter must not be incremented
        self.assertEqual(state.gws.read_counter(3), 0)

    def test_registers_preserved_across_all_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            for r in range(10):
                state.set_lane_vgpr(lane, r, (lane << 16) | r)

        instr = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=4)
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

        state.set_vgpr(0, 0x11223344)
        state.set_vgpr(255, 0x55667788)
        state.set_m0(10)

        v0 = RegisterOperand(0, RegType.VGPR)
        instr_v0 = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, vdata0=v0, offset=1)
        interp.execute(instr_v0, state)
        self.assertEqual(state.get_vgpr(0), 0x11223344)
        self.assertEqual(state.get_m0(), 10)

        v255 = RegisterOperand(255, RegType.VGPR)
        instr_v255 = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, vdata0=v255, offset=1)
        interp.execute(instr_v255, state, wave_id=1)
        self.assertEqual(state.get_vgpr(255), 0x55667788)
        self.assertEqual(state.get_m0(), 10)

    def test_scalar_full_register_file_undisturbed(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        for i in range(DEFAULT_NUM_VGPRS):
            state.set_vgpr(i, 0xAA000000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            state.set_sgpr(i, 0xBB000000 | i)
        state.set_m0(0x25)

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=5)
        instr_p = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=5)

        interp.execute(instr_v, state)
        interp.execute(instr_p, state, wave_id=1)

        for i in range(DEFAULT_NUM_VGPRS):
            self.assertEqual(state.get_vgpr(i), 0xAA000000 | i)
        for i in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(state.get_sgpr(i), 0xBB000000 | i)
        self.assertEqual(state.get_m0(), 0x25)

    def test_scalar_repeated_operand_use_across_semas(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        common_val = 0xCAFEBABE
        state.set_vgpr(8, common_val)
        v8 = RegisterOperand(8, RegType.VGPR)

        for i in range(5):
            instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, vdata0=v8, offset=i)
            interp.execute(instr_v, state)
            self.assertEqual(state.get_vgpr(8), common_val)

        for i in range(5):
            self.assertEqual(state.gws.read_counter(i), 1)

    def test_simd_multi_lane_preservation_under_divergence(self) -> None:
        mask = 0x5A5A5A5A5A5A5A5A
        state = SIMDState(num_lanes=64, exec_mask=mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            for r in range(16):
                state.set_lane_vgpr(lane, r, ((lane + 1) * 300) + r)

        instr = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=12)
        res = interp.execute(instr, state)
        self.assertIsNotNone(res)

        for lane in range(64):
            for r in range(16):
                expected = ((lane + 1) * 300) + r
                self.assertEqual(state.get_lane_vgpr(lane, r), expected)


class TestConditionCodesStatusFlagsAndCarryBorrow(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_preserved_scalar(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vcc = 0xDEADBEEF_01234567
        state.scc = 1

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=0)
        interp.execute(instr_v, state)
        self.assertEqual(state.vcc, 0xDEADBEEF_01234567)
        self.assertEqual(state.scc, 1)

        instr_p = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=0)
        interp.execute(instr_p, state, wave_id=1)
        self.assertEqual(state.vcc, 0xDEADBEEF_01234567)
        self.assertEqual(state.scc, 1)

    def test_vcc_and_scc_preserved_simd(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        state.vcc = 0x88776655
        state.scc = 0

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=3)
        interp.execute(instr_v, state)
        self.assertEqual(state.vcc, 0x88776655)
        self.assertEqual(state.scc, 0)

    def test_no_carry_borrow_generated(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.vcc = 0
        state.scc = 0

        instr = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=63)
        interp.execute(instr, state, wave_id=1)

        self.assertEqual(state.vcc, 0)
        self.assertEqual(state.scc, 0)

    def test_status_flags_semaphore_availability(self) -> None:
        # Counter 0: not available, counter_zero=True
        st0 = evaluate_gws_sema_status(counter=0)
        self.assertFalse(st0["semaphore_available"])
        self.assertTrue(st0["counter_zero"])

        # Counter 1: available, counter_zero=False
        st1 = evaluate_gws_sema_status(counter=1)
        self.assertTrue(st1["semaphore_available"])
        self.assertFalse(st1["counter_zero"])

    def test_status_flags_offset_carry_out(self) -> None:
        st_no_carry = evaluate_gws_sema_status(counter=1, offset=10, m0=20)
        self.assertFalse(st_no_carry["offset_carry_out_16"])
        self.assertFalse(st_no_carry["offset_carry_out_32"])

        st_carry_16 = evaluate_gws_sema_status(counter=1, offset=0x0020, m0=0xFFF0)
        self.assertTrue(st_carry_16["offset_carry_out_16"])
        self.assertFalse(st_carry_16["offset_carry_out_32"])

        st_carry_32 = evaluate_gws_sema_status(counter=1, offset=1, m0=0xFFFFFFFF)
        self.assertTrue(st_carry_32["offset_carry_out_32"])

    def test_status_flags_wave_activity(self) -> None:
        st_active = evaluate_gws_sema_status(counter=1, exec_mask=0x1)
        self.assertTrue(st_active["active_wave"])

        st_inactive = evaluate_gws_sema_status(counter=1, exec_mask=0x0)
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

        state.set_vgpr(0, self.F32_POS_ZERO)
        state.set_vgpr(1, self.F32_NEG_ZERO)

        instr_v = DSGWSSemaphoreInstruction(
            opcode=DSOpcode.DS_GWS_SEMA_V,
            vdata0=RegisterOperand(0, RegType.VGPR),
            offset=0,
        )
        interp.execute(instr_v, state)
        self.assertEqual(state.get_vgpr(0), self.F32_POS_ZERO)

        instr_p = DSGWSSemaphoreInstruction(
            opcode=DSOpcode.DS_GWS_SEMA_P,
            vdata0=RegisterOperand(1, RegType.VGPR),
            offset=0,
        )
        interp.execute(instr_p, state, wave_id=1)
        self.assertEqual(state.get_vgpr(1), self.F32_NEG_ZERO)

    def test_subnormals_preservation_no_ftz(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, self.F32_MIN_SUBNORMAL)
        instr = DSGWSSemaphoreInstruction(
            opcode=DSOpcode.DS_GWS_SEMA_V,
            vdata0=RegisterOperand(0, RegType.VGPR),
            offset=1,
        )
        interp.execute(instr, state)
        self.assertEqual(state.get_vgpr(0), self.F32_MIN_SUBNORMAL)

    def test_normals_and_infinities_preservation(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        patterns = [
            self.F32_MIN_NORMAL,
            self.F32_MAX_NORMAL,
            self.F32_POS_INF,
            self.F32_NEG_INF,
        ]

        for reg_idx, pattern in enumerate(patterns):
            state.set_vgpr(reg_idx, pattern)
            instr = DSGWSSemaphoreInstruction(
                opcode=DSOpcode.DS_GWS_SEMA_V,
                vdata0=RegisterOperand(reg_idx, RegType.VGPR),
                offset=reg_idx,
            )
            interp.execute(instr, state)
            self.assertEqual(state.get_vgpr(reg_idx), pattern)

    def test_nans_payload_preservation_no_quieting(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        nan_patterns = [
            (self.F32_QNAN_CANONICAL, 10),
            (self.F32_SNAN, 11),
            (self.F32_NEG_QNAN, 12),
            (self.F32_CUSTOM_NAN_PAYLOAD, 13),
        ]

        for pattern, reg_idx in nan_patterns:
            state.set_vgpr(reg_idx, pattern)
            instr = DSGWSSemaphoreInstruction(
                opcode=DSOpcode.DS_GWS_SEMA_V,
                vdata0=RegisterOperand(reg_idx, RegType.VGPR),
                offset=reg_idx,
            )
            interp.execute(instr, state)
            self.assertEqual(state.get_vgpr(reg_idx), pattern)
            if pattern == self.F32_SNAN:
                self.assertEqual(state.get_vgpr(reg_idx) & 0x0040_0000, 0)

    def test_packed_f16_bit_patterns(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        state.set_vgpr(0, self.F16_PACKED_ONES)
        interp.execute(
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, vdata0=RegisterOperand(0, RegType.VGPR), offset=20),
            state,
        )
        self.assertEqual(state.get_vgpr(0), self.F16_PACKED_ONES)

    def test_simd_nan_preservation_across_all_lanes(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.set_lane_vgpr(lane, 0, self.F32_CUSTOM_NAN_PAYLOAD)

        instr = DSGWSSemaphoreInstruction(
            opcode=DSOpcode.DS_GWS_SEMA_V,
            vdata0=RegisterOperand(0, RegType.VGPR),
            offset=30,
        )
        interp.execute(instr, state)

        for lane in range(32):
            self.assertEqual(state.get_lane_vgpr(lane, 0), self.F32_CUSTOM_NAN_PAYLOAD)


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
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, vdata0=s0, offset=0, gds=True)

    def test_syntax_rejection_invalid_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=-1, gds=True)

        with self.assertRaises(ValueError):
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=0x10000, gds=True)

    def test_syntax_rejection_missing_mandatory_gds(self) -> None:
        with self.assertRaises(ValueError):
            DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_P, offset=0, gds=False)

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
            GWSMemory(num_resources=48)

    def test_mutation_detection_sema_p_underflow_when_zero(self) -> None:
        # Reference: counter=0 -> acq=False, counter remains 0
        acq_ref, cnt_ref = ref_ds_gws_sema_p(0)
        self.assertFalse(acq_ref)
        self.assertEqual(cnt_ref, 0)

        # Buggy mutant decrements without checking > 0, producing 0xFFFFFFFF
        mutant_cnt = u32(0 - 1)
        self.assertNotEqual(mutant_cnt, cnt_ref)
        self.assertEqual(mutant_cnt, 0xFFFFFFFF)

    def test_mutation_detection_sema_v_decrement_instead_of_increment(self) -> None:
        ref_cnt = ref_ds_gws_sema_v(5)
        self.assertEqual(ref_cnt, 6)

        mutant_cnt = u32(5 - 1)
        self.assertNotEqual(mutant_cnt, ref_cnt)

    def test_mutation_detection_exec_zero_bypass(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=8)
        res = interp.execute(instr_v, state)

        self.assertIsNone(res)
        self.assertEqual(state.gws.read_counter(8), 0)

        mutant_executed_cnt = 1
        self.assertNotEqual(mutant_executed_cnt, state.gws.read_counter(8))

    def test_mutation_detection_dropped_offset(self) -> None:
        state = ScalarState()
        interp = ScalarInterpreter()

        target_offset = 35
        instr_v = DSGWSSemaphoreInstruction(opcode=DSOpcode.DS_GWS_SEMA_V, offset=target_offset)

        rid, cnt, _ = interp.execute(instr_v, state)
        self.assertEqual(rid, target_offset)
        self.assertEqual(state.gws.read_counter(target_offset), 1)
        self.assertEqual(state.gws.read_counter(0), 0)

        mutant_rid = 0
        self.assertNotEqual(mutant_rid, rid)

    def test_mutation_detection_fifo_wake_order(self) -> None:
        gws = GWSMemory()
        # Waves 101 and 102 queue up on resource 0
        gws.sema_p(offset=0, wave_id=101)
        gws.sema_p(offset=0, wave_id=102)

        # Release must wake wave 101 first (FIFO)
        _, _, woken1 = gws.sema_v(offset=0)
        self.assertEqual(woken1, 101)

        # Flawed LIFO mutant would have woken wave 102
        mutant_lifo_woken = 102
        self.assertNotEqual(mutant_lifo_woken, woken1)


if __name__ == "__main__":
    unittest.main()
