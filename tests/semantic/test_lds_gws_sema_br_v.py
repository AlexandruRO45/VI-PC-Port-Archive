from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 ds_gws_sema_br_v.

Models AMD RDNA2 / GFX10 Global Wavefront Sync (GWS) hardware semaphore
branch-release operation (bulk signal with branch / wakeup notification):
  - ds_gws_sema_br_v: Releases permits to the targeted GWS semaphore resource.
    Awakens waiting wavefronts from the FIFO wait queue up to the released
    permit count. Any unconsumed permits increment the resource counter.
    Signals branch taken / SCC=1 if one or more wavefronts were awakened.
    Signals branch not taken / SCC=0 if no wavefronts were waiting.

RDNA2 GWS Architectural Rules:
  1. GWS Resource Addressing:
     - 64 hardware resource slots (0..63) in GDS space.
     - Effective target resource ID: (m0 + (offset16 & 0xFFFF)) & 0x3F.
  2. Semaphore Branch Release Semantics:
     - Release Count: Specified via vdata (VGPR) or defaults to 1 permit.
     - Waiter Awakening: Waiting wavefronts are dequeued in strict FIFO order.
     - Counter Accumulation: Unconsumed permits add to the 32-bit unsigned counter.
     - Branch Condition: Evaluates to True (SCC=1) if waiters were awakened.
  3. Resource Isolation & Mode Enforcement:
     - Each resource slot is independently configured (UNINITIALIZED, SEMAPHORE, BARRIER).
     - Executing ds_gws_sema_br_v on a BARRIER resource triggers an architectural type fault.

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
    wake_count: int = 0

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
        res.wake_count = 0

    def bulk_release_br_v(self, res_id: int, release_count: int = 1) -> Tuple[int, List[int], bool]:
        """Executes a semaphore branch release (bulk signal with branch indicator).

        Args:
            res_id: GWS resource index [0..63].
            release_count: Number of permits delivered in the bulk release.

        Returns:
            Tuple of (new_counter_value, list_of_awakened_wavefront_ids, branch_taken).
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_br_v")
        res.res_type = GWSResourceType.SEMAPHORE

        count = max(1, release_count & MASK32)
        res.signal_count += count
        awakened: List[int] = []

        if res.wait_queue:
            num_to_wake = min(count, len(res.wait_queue))
            awakened = res.wait_queue[:num_to_wake]
            res.wait_queue = res.wait_queue[num_to_wake:]
            res.wake_count += len(awakened)
            remaining_permits = count - num_to_wake
            res.value = (res.value + remaining_permits) & MASK32
        else:
            res.value = (res.value + count) & MASK32

        branch_taken = (len(awakened) > 0)
        return res.value, awakened, branch_taken

    def wait_p(self, res_id: int, wave_id: int) -> bool:
        """Executes a semaphore P (wait/decrement/acquire) operation."""
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_p")
        res.res_type = GWSResourceType.SEMAPHORE

        if res.value > 0:
            res.value = (res.value - 1) & MASK32
            return True
        else:
            res.wait_queue.append(wave_id)
            return False


def ref_gws_resource_id(m0_val: int, offset: int) -> int:
    """Calculates target GWS resource index from M0 base register and offset."""
    base = m0_val & GWS_RESOURCE_ID_MASK
    return (base + (offset & OFFSET16_MASK)) & GWS_RESOURCE_ID_MASK


def ref_ds_gws_sema_br_v(subsystem: GWSSubsystem, m0_val: int, offset: int, release_count: int = 1) -> Tuple[int, List[int], bool]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_br_v."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.bulk_release_br_v(rid, release_count=release_count)


# ============================================================================
# Commit 2: Symbolic AST opcode definitions and typed expressions
# ============================================================================


class RegKind(enum.Enum):
    """Register kinds in RDNA2 architecture."""
    VGPR = "vgpr"
    SGPR = "sgpr"
    M0 = "m0"
    EXEC = "exec"
    VCC = "vcc"
    SCC = "scc"
    STATUS = "status"


@dataclass(frozen=True)
class Register:
    """Represents an RDNA2 hardware register."""
    index: int = 0
    kind: RegKind = RegKind.VGPR

    def __post_init__(self) -> None:
        if self.kind in (RegKind.VGPR, RegKind.SGPR):
            if not (0 <= self.index <= 255):
                raise ValueError(f"Register index out of bounds [0, 255]: {self.index}")
        elif self.kind in (RegKind.M0, RegKind.EXEC, RegKind.VCC, RegKind.SCC, RegKind.STATUS):
            if self.index != 0:
                raise ValueError(f"Special register {self.kind.value} index must be 0")

    def __str__(self) -> str:
        if self.kind == RegKind.VGPR:
            return f"v{self.index}"
        elif self.kind == RegKind.SGPR:
            return f"s{self.index}"
        return self.kind.value


class Opcode(enum.Enum):
    """Data Share / GWS Opcode definitions."""
    DS_GWS_SEMA_BR_V = "ds_gws_sema_br_v"
    DS_GWS_SEMA_BR_P = "ds_gws_sema_br_p"
    DS_GWS_SEMA_BR = "ds_gws_sema_br"
    DS_GWS_SEMA_P = "ds_gws_sema_p"
    DS_GWS_SEMA_V = "ds_gws_sema_v"
    DS_GWS_INIT = "ds_gws_init"
    DS_GWS_BARRIER = "ds_gws_barrier"


@dataclass(frozen=True)
class Instruction:
    """AST representation of RDNA2 ds_gws_sema_br_v instruction."""
    opcode: Opcode = Opcode.DS_GWS_SEMA_BR_V
    offset: int = 0
    gds: bool = True
    vdata: Optional[Register] = None

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"GWS immediate offset out of range [0, 65535]: {self.offset}")
        if not self.gds:
            raise ValueError("GWS instructions strictly require the gds keyword modifier")
        if self.vdata is not None and self.vdata.kind != RegKind.VGPR:
            raise TypeError(f"vdata must be a VGPR, got: {self.vdata.kind}")

    def disasm(self) -> str:
        parts = [self.opcode.value]
        if self.vdata is not None:
            parts.append(str(self.vdata))
        parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)

    @classmethod
    def parse(cls, asm: str) -> Instruction:
        """Parse assembly text into Instruction AST."""
        tokens = asm.strip().split()
        if not tokens:
            raise ValueError("Empty assembly string")

        op_str = tokens[0]
        try:
            opcode = Opcode(op_str)
        except ValueError:
            raise ValueError(f"Unrecognized opcode: {op_str}")

        if tokens[-1] != "gds":
            raise ValueError("Instruction must end with 'gds' modifier")
        gds = True

        offset = 0
        vdata = None

        for tok in tokens[1:-1]:
            if tok.startswith("offset:"):
                raw_off = tok.split(":", 1)[1]
                offset = int(raw_off, 0)
            elif tok.startswith("v") and tok[1:].isdigit():
                vdata = Register(int(tok[1:]), RegKind.VGPR)
            else:
                raise ValueError(f"Unexpected token in instruction: {tok}")

        return cls(opcode=opcode, offset=offset, gds=gds, vdata=vdata)


@dataclass(frozen=True)
class GWSResourceExpr:
    """Symbolic expression for effective GWS resource identifier."""
    base_reg: Register = field(default_factory=lambda: Register(0, RegKind.M0))
    offset: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"Offset out of range [0, 65535]: {self.offset}")

    def evaluate(self, m0_val: int) -> int:
        return ref_gws_resource_id(m0_val, self.offset)


@dataclass(frozen=True)
class SemaphoreBranchReleaseExpr:
    """Symbolic typed expression for semaphore branch release operation."""
    resource_id: int

    def calculate_new_value(self, current_val: int, release_count: int, num_waiters: int) -> int:
        count = max(1, release_count & MASK32)
        unconsumed = max(0, count - num_waiters)
        return (current_val + unconsumed) & MASK32

    def num_awakened(self, release_count: int, num_waiters: int) -> int:
        count = max(1, release_count & MASK32)
        return min(count, num_waiters)

    def branch_taken(self, num_waiters: int) -> bool:
        return num_waiters > 0


# ============================================================================
# Commit 3: Deterministic scalar instruction interpreter
# ============================================================================


class ThreadExecutionStatus(enum.Enum):
    """Thread / Wavefront execution lifecycle status."""
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"


@dataclass
class ScalarThreadState:
    """Models the execution state of an individual scalar thread in RDNA2."""
    wave_id: int = 0
    m0: int = 0
    sgprs: Dict[int, int] = field(default_factory=dict)
    vgprs: Dict[int, int] = field(default_factory=dict)
    exec_mask: int = 1
    status: ThreadExecutionStatus = ThreadExecutionStatus.RUNNING
    scc: int = 0
    branch_target: Optional[int] = None
    gws: GWSSubsystem = field(default_factory=GWSSubsystem)

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx <= 255):
            raise ValueError(f"SGPR index out of bounds: {idx}")
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx <= 255):
            raise ValueError(f"SGPR index out of bounds: {idx}")
        self.sgprs[idx] = val & MASK32

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index out of bounds: {idx}")
        return self.vgprs.get(idx, 0)

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index out of bounds: {idx}")
        self.vgprs[idx] = val & MASK32

    def set_m0(self, val: int) -> None:
        self.m0 = val & MASK32


class ScalarGWSInterpreter:
    """Deterministic scalar instruction interpreter for GWS branch release instructions."""

    def execute(
        self,
        state: ScalarThreadState,
        inst: Instruction,
        branch_pc: Optional[int] = None,
    ) -> Tuple[int, List[int], bool]:
        """Execute a single GWS branch release instruction deterministically.

        Returns:
            Tuple[int, List[int], bool]: (new_counter_val, awakened_wavefronts, branch_taken)
        """
        if state.status == ThreadExecutionStatus.BLOCKED:
            raise RuntimeError(f"Cannot execute instruction on suspended wavefront {state.wave_id}")

        if inst.opcode != Opcode.DS_GWS_SEMA_BR_V:
            raise ValueError(f"Unsupported opcode in ScalarGWSInterpreter: {inst.opcode}")

        if (state.exec_mask & 1) == 0:
            rid = ref_gws_resource_id(state.m0, inst.offset)
            current_val = state.gws.resources[rid].value
            return current_val, [], False

        rid = ref_gws_resource_id(state.m0, inst.offset)

        count = 1
        if inst.vdata is not None:
            count = max(1, state.get_vgpr(inst.vdata.index))

        val, awakened, branch_taken = state.gws.bulk_release_br_v(rid, release_count=count)

        if branch_taken:
            state.scc = 1
            state.branch_target = branch_pc
        else:
            state.scc = 0
            state.branch_target = None

        state.status = ThreadExecutionStatus.RUNNING
        return val, awakened, branch_taken


# ============================================================================
# Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking
# ============================================================================


@dataclass
class SIMDWaveState:
    """Models an AMD RDNA2 wavefront (Wave32 or Wave64) executing in SIMD."""
    wave_id: int = 0
    wave_size: int = 32
    exec_mask: int = 0xFFFF_FFFF
    m0: int = 0
    sgprs: Dict[int, int] = field(default_factory=dict)
    lane_vgprs: List[Dict[int, int]] = field(default_factory=list)
    status: ThreadExecutionStatus = ThreadExecutionStatus.RUNNING
    scc: int = 0
    branch_target: Optional[int] = None
    gws: GWSSubsystem = field(default_factory=GWSSubsystem)

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave_size {self.wave_size}; must be 32 or 64")
        if self.wave_size == 32:
            self.exec_mask &= MASK32
        else:
            self.exec_mask &= MASK64
        if not self.lane_vgprs:
            self.lane_vgprs = [dict() for _ in range(self.wave_size)]
        elif len(self.lane_vgprs) != self.wave_size:
            raise ValueError(f"lane_vgprs length {len(self.lane_vgprs)} != wave_size {self.wave_size}")

    def get_vgpr(self, lane: int, idx: int) -> int:
        if not (0 <= lane < self.wave_size):
            raise ValueError(f"Lane {lane} out of bounds for Wave{self.wave_size}")
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index {idx} out of bounds")
        return self.lane_vgprs[lane].get(idx, 0)

    def set_vgpr(self, lane: int, idx: int, val: int) -> None:
        if not (0 <= lane < self.wave_size):
            raise ValueError(f"Lane {lane} out of bounds for Wave{self.wave_size}")
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index {idx} out of bounds")
        self.lane_vgprs[lane][idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx <= 255):
            raise ValueError(f"SGPR index {idx} out of bounds")
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx <= 255):
            raise ValueError(f"SGPR index {idx} out of bounds")
        self.sgprs[idx] = val & MASK32

    def first_active_lane(self) -> Optional[int]:
        """Returns the index of the first active lane in the execution mask."""
        mask = self.exec_mask & ((1 << self.wave_size) - 1)
        if mask == 0:
            return None
        return (mask & -mask).bit_length() - 1


class SIMDGWSInterpreter:
    """SIMD wavefront instruction interpreter with 32/64-bit EXEC divergence."""

    def execute(
        self,
        wave: SIMDWaveState,
        inst: Instruction,
        branch_pc: Optional[int] = None,
    ) -> Tuple[int, List[int], bool]:
        """Execute a GWS branch release on a full wavefront under EXEC mask.

        Returns:
            Tuple[int, List[int], bool]: (new_counter_val, awakened_wavefronts, branch_taken)
        """
        if wave.status == ThreadExecutionStatus.BLOCKED:
            raise RuntimeError(f"Cannot execute instruction on suspended wavefront {wave.wave_id}")

        if inst.opcode != Opcode.DS_GWS_SEMA_BR_V:
            raise ValueError(f"Unsupported opcode in SIMDGWSInterpreter: {inst.opcode}")

        active_lane = wave.first_active_lane()
        if active_lane is None:
            rid = ref_gws_resource_id(wave.m0, inst.offset)
            return wave.gws.resources[rid].value, [], False

        rid = ref_gws_resource_id(wave.m0, inst.offset)

        count = 1
        if inst.vdata is not None:
            count = max(1, wave.get_vgpr(active_lane, inst.vdata.index))

        val, awakened, branch_taken = wave.gws.bulk_release_br_v(rid, release_count=count)

        if branch_taken:
            wave.scc = 1
            wave.branch_target = branch_pc
        else:
            wave.scc = 0
            wave.branch_target = None

        wave.status = ThreadExecutionStatus.RUNNING
        return val, awakened, branch_taken


# ============================================================================
# Commit 5: Register aliasing and in-place operand preservation
# ============================================================================


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_br_v."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_release_increments_counter(self) -> None:
        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=0, offset=0, release_count=1)
        self.assertEqual(val, 1)
        self.assertEqual(awakened, [])
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[0].value, 1)

    def test_multi_permit_release_accumulates(self) -> None:
        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=2, offset=3, release_count=16)
        self.assertEqual(val, 16)
        self.assertEqual(awakened, [])
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[5].value, 16)

    def test_release_wakes_waiting_wavefronts_fifo(self) -> None:
        rid = 8
        for w in [10, 20, 30]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=3)
        self.assertEqual(awakened, [10, 20, 30])
        self.assertEqual(val, 0)
        self.assertTrue(branch)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_release_partial_waiters(self) -> None:
        rid = 10
        for w in [1, 2, 3, 4, 5]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=2)
        self.assertEqual(awakened, [1, 2])
        self.assertEqual(val, 0)
        self.assertTrue(branch)
        self.assertEqual(self.gws.resources[rid].wait_queue, [3, 4, 5])

    def test_release_exceeding_waiters_accumulates_remainder(self) -> None:
        rid = 12
        for w in [101, 102]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened, branch = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=5)
        self.assertEqual(awakened, [101, 102])
        self.assertEqual(val, 3)
        self.assertTrue(branch)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_counter_wrapping_modulo_32bit(self) -> None:
        rid = 15
        self.gws.init_resource(rid, init_val=0xFFFF_FFF0)
        val, _, _ = ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=20)
        self.assertEqual(val, 4)
        self.assertEqual(self.gws.resources[rid].value, 4)

    def test_resource_id_modulo_64_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)
        self.assertEqual(ref_gws_resource_id(m0_val=0, offset=0x10005), 5)

    def test_resource_isolation(self) -> None:
        ref_ds_gws_sema_br_v(self.gws, m0_val=1, offset=0, release_count=5)
        ref_ds_gws_sema_br_v(self.gws, m0_val=2, offset=0, release_count=10)

        self.assertEqual(self.gws.resources[1].value, 5)
        self.assertEqual(self.gws.resources[2].value, 10)
        self.assertEqual(self.gws.resources[0].value, 0)

    def test_barrier_mode_rejection(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=0, res_type=GWSResourceType.BARRIER)
        with self.assertRaises(TypeError):
            ref_ds_gws_sema_br_v(self.gws, m0_val=rid, offset=0, release_count=1)


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions for ds_gws_sema_br_v."""

    def test_register_construction_and_formatting(self) -> None:
        v0 = Register(0, RegKind.VGPR)
        v255 = Register(255, RegKind.VGPR)
        s15 = Register(15, RegKind.SGPR)
        m0 = Register(0, RegKind.M0)
        exec_reg = Register(0, RegKind.EXEC)

        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        self.assertEqual(str(s15), "s15")
        self.assertEqual(str(m0), "m0")
        self.assertEqual(str(exec_reg), "exec")

    def test_register_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Register(-1, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(256, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(1, RegKind.M0)

    def test_instruction_disassembly(self) -> None:
        i1 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)
        self.assertEqual(i1.disasm(), "ds_gws_sema_br_v offset:0 gds")

        i2 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=12, gds=True)
        self.assertEqual(i2.disasm(), "ds_gws_sema_br_v offset:12 gds")

        i3 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=4, gds=True, vdata=Register(3, RegKind.VGPR))
        self.assertEqual(i3.disasm(), "ds_gws_sema_br_v v3 offset:4 gds")

    def test_instruction_assembly_parser(self) -> None:
        i1 = Instruction.parse("ds_gws_sema_br_v offset:0 gds")
        self.assertEqual(i1.opcode, Opcode.DS_GWS_SEMA_BR_V)
        self.assertEqual(i1.offset, 0)
        self.assertTrue(i1.gds)

        i2 = Instruction.parse("ds_gws_sema_br_v v2 offset:8 gds")
        self.assertIsNotNone(i2.vdata)
        self.assertEqual(i2.vdata.index, 2)
        self.assertEqual(i2.offset, 8)

    def test_instruction_parser_rejections(self) -> None:
        with self.assertRaises(ValueError):
            Instruction.parse("ds_gws_sema_br_v offset:0")
        with self.assertRaises(ValueError):
            Instruction.parse("ds_gws_sema_br_v gds offset:0")
        with self.assertRaises(ValueError):
            Instruction.parse("v_add_f32 v0, v1, v2")

    def test_gws_resource_expr_evaluation(self) -> None:
        expr = GWSResourceExpr(offset=10)
        self.assertEqual(expr.evaluate(m0_val=2), 12)
        self.assertEqual(expr.evaluate(m0_val=60), 6)

    def test_semaphore_branch_release_expr(self) -> None:
        expr = SemaphoreBranchReleaseExpr(resource_id=5)
        # Release 4 permits when 2 waiters present: 2 awakened, 2 added to counter 10 -> 12
        self.assertEqual(expr.calculate_new_value(current_val=10, release_count=4, num_waiters=2), 12)
        self.assertEqual(expr.num_awakened(release_count=4, num_waiters=2), 2)
        self.assertTrue(expr.branch_taken(num_waiters=2))

        # Release 3 permits when 0 waiters present: 0 awakened, 3 added to counter 0 -> 3
        self.assertEqual(expr.calculate_new_value(current_val=0, release_count=3, num_waiters=0), 3)
        self.assertEqual(expr.num_awakened(release_count=3, num_waiters=0), 0)
        self.assertFalse(expr.branch_taken(num_waiters=0))


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter for ds_gws_sema_br_v."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.interp = ScalarGWSInterpreter()

    def test_scalar_release_without_waiters(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(state, inst, branch_pc=0x3000)
        self.assertEqual(val, 1)
        self.assertEqual(awakened, [])
        self.assertFalse(branch)
        self.assertEqual(state.scc, 0)
        self.assertIsNone(state.branch_target)

    def test_scalar_release_with_waiters(self) -> None:
        rid = 6
        self.gws.init_resource(rid, init_val=0)
        self.gws.wait_p(rid, wave_id=10)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(state, inst, branch_pc=0x4000)
        self.assertEqual(val, 0)
        self.assertEqual(awakened, [10])
        self.assertTrue(branch)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.branch_target, 0x4000)

    def test_scalar_release_with_vdata_permits(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=0)
        for w in [101, 102, 103]:
            self.gws.wait_p(rid, wave_id=w)

        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_vgpr(4, 5)  # release 5 permits
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True, vdata=Register(4, RegKind.VGPR))

        val, awakened, branch = self.interp.execute(state, inst)
        self.assertEqual(awakened, [101, 102, 103])
        self.assertEqual(val, 2)  # 2 remaining permits added to counter
        self.assertTrue(branch)
        self.assertEqual(state.scc, 1)

    def test_scalar_release_inactive_exec(self) -> None:
        rid = 8
        self.gws.init_resource(rid, init_val=5)
        state = ScalarThreadState(wave_id=1, m0=rid, exec_mask=0, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(state, inst)
        self.assertEqual(val, 5)
        self.assertEqual(awakened, [])
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[rid].value, 5)

    def test_dynamic_m0_offset_resolution(self) -> None:
        self.gws.init_resource(14, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=10, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=4, gds=True)

        val, awakened, branch = self.interp.execute(state, inst)
        self.assertEqual(val, 1)
        self.assertEqual(self.gws.resources[14].value, 1)


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for ds_gws_sema_br_v."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.interp = SIMDGWSInterpreter()

    def test_wave32_full_exec_single_release(self) -> None:
        rid = 1
        self.gws.init_resource(rid, init_val=0)
        wave = SIMDWaveState(wave_id=1, wave_size=32, exec_mask=0xFFFF_FFFF, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(wave, inst)
        # Exactly 1 permit added for the wavefront, not 32!
        self.assertEqual(val, 1)
        self.assertEqual(awakened, [])
        self.assertFalse(branch)

    def test_wave32_divergent_mask_single_lane(self) -> None:
        rid = 2
        self.gws.init_resource(rid, init_val=0)
        wave = SIMDWaveState(wave_id=2, wave_size=32, exec_mask=(1 << 21), m0=rid, gws=self.gws)
        self.assertEqual(wave.first_active_lane(), 21)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(wave, inst)
        self.assertEqual(val, 1)

    def test_wave32_divergent_vdata_permits(self) -> None:
        rid = 3
        self.gws.init_resource(rid, init_val=0)
        wave = SIMDWaveState(wave_id=3, wave_size=32, exec_mask=(1 << 2) | (1 << 5), m0=rid, gws=self.gws)
        wave.set_vgpr(2, 1, 6)   # First active lane requests 6
        wave.set_vgpr(5, 1, 99)  # Higher lane should be ignored

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True, vdata=Register(1, RegKind.VGPR))
        val, awakened, branch = self.interp.execute(wave, inst)
        self.assertEqual(val, 6)

    def test_wave32_exec_zero_suppression(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=7)
        wave = SIMDWaveState(wave_id=4, wave_size=32, exec_mask=0, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(wave, inst)
        self.assertEqual(val, 7)
        self.assertEqual(self.gws.resources[rid].value, 7)

    def test_wave64_full_exec_single_release(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=0)
        wave = SIMDWaveState(wave_id=10, wave_size=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(wave, inst)
        self.assertEqual(val, 1)

    def test_wave64_upper_lanes_divergence(self) -> None:
        rid = 6
        self.gws.init_resource(rid, init_val=0)
        wave = SIMDWaveState(wave_id=11, wave_size=64, exec_mask=0xFFFF_0000_0000_0000, m0=rid, gws=self.gws)
        self.assertEqual(wave.first_active_lane(), 48)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(wave, inst)
        self.assertEqual(val, 1)

    def test_wave_awakening_via_simd_release(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=0)
        self.gws.wait_p(rid, wave_id=99)

        wave = SIMDWaveState(wave_id=20, wave_size=32, exec_mask=0xFFFF_FFFF, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        val, awakened, branch = self.interp.execute(wave, inst, branch_pc=0x9000)
        self.assertEqual(awakened, [99])
        self.assertEqual(val, 0)
        self.assertTrue(branch)
        self.assertEqual(wave.scc, 1)
        self.assertEqual(wave.branch_target, 0x9000)


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation for ds_gws_sema_br_v."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.scalar_interp = ScalarGWSInterpreter()
        self.simd_interp = SIMDGWSInterpreter()

    def test_m0_preservation_empty_queue(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        self.scalar_interp.execute(state, inst)
        self.assertEqual(state.m0, rid)

    def test_m0_preservation_with_waiters(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=0)
        self.gws.wait_p(rid, wave_id=10)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)

        self.scalar_interp.execute(state, inst)
        self.assertEqual(state.m0, rid)

    def test_all_sgprs_preserved_in_place(self) -> None:
        rid = 9
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        for s in range(106):
            state.set_sgpr(s, (0xDEAD0000 + s * 43) & MASK32)
        saved_sgprs = dict(state.sgprs)

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)
        self.scalar_interp.execute(state, inst)

        self.assertEqual(state.sgprs, saved_sgprs)

    def test_all_lane_vgprs_preserved_in_simd(self) -> None:
        rid = 12
        self.gws.init_resource(rid, init_val=0)
        wave = SIMDWaveState(wave_id=5, wave_size=32, m0=rid, gws=self.gws)
        for lane in range(32):
            for v in range(0, 256, 16):
                wave.set_vgpr(lane, v, (lane << 16) | (v * 11))
        saved_vgprs = [dict(lane_dict) for lane_dict in wave.lane_vgprs]

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)
        self.simd_interp.execute(wave, inst)

        self.assertEqual(wave.lane_vgprs, saved_vgprs)

    def test_vdata_operand_preservation(self) -> None:
        rid = 15
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_vgpr(5, 7)  # release 7 permits

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True, vdata=Register(5, RegKind.VGPR))
        val, awakened, branch = self.scalar_interp.execute(state, inst)
        self.assertEqual(val, 7)
        self.assertEqual(state.get_vgpr(5), 7)

    def test_aliased_m0_and_sgpr_integrity(self) -> None:
        rid = 18
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_sgpr(0, rid)
        state.set_sgpr(1, rid)

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)
        self.scalar_interp.execute(state, inst)

        self.assertEqual(state.m0, rid)
        self.assertEqual(state.get_sgpr(0), rid)
        self.assertEqual(state.get_sgpr(1), rid)

    def test_consecutive_release_pipeline_preservation(self) -> None:
        rid = 20
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_sgpr(10, 0x11223344)
        state.set_vgpr(20, 0x55667788)

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_V, offset=0, gds=True)
        for _ in range(5):
            self.scalar_interp.execute(state, inst)
            self.assertEqual(state.m0, rid)
            self.assertEqual(state.get_sgpr(10), 0x11223344)
            self.assertEqual(state.get_vgpr(20), 0x55667788)


if __name__ == "__main__":
    unittest.main()
