from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 ds_gws_sema_p.

Models the AMD RDNA2 / GFX10 Global Wavefront Sync (GWS) hardware semaphore
P-operation (acquire / wait / decrement):
  - ds_gws_sema_p: Decrements the target GWS semaphore counter if > 0.
    If the counter is 0, suspends/blocks the calling wavefront and enqueues
    it in the GWS hardware FIFO wait queue until awakened by ds_gws_sema_v.

RDNA2 GWS Architectural Rules:
  1. GWS Resource Addressing:
     - 64 hardware resource slots (0..63) in GDS space.
     - Target resource ID: (m0 + (offset16 & 0xFFFF)) & 0x3F.
  2. Semaphore P Semantics (Dijkstra Proberen / Wait / Down):
     - If counter > 0: counter is decremented by 1 (counter -= 1); acquire succeeds immediately.
     - If counter == 0: wavefront suspends and is enqueued into the resource FIFO wait queue.
  3. Resource Isolation & Mode Enforcement:
     - Each resource slot is independently configured (UNINITIALIZED, SEMAPHORE, BARRIER).
     - Executing ds_gws_sema_p on a BARRIER resource triggers an architectural type fault.

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

    def wait_p(self, res_id: int, wave_id: int) -> Tuple[bool, int, bool]:
        """Executes a semaphore P (wait/decrement/acquire) operation.

        Returns:
            Tuple[bool, int, bool]: (acquired, new_counter_value, blocked)
        """
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
        """Executes a semaphore V (signal/increment/post) operation.

        Returns:
            Tuple[int, Optional[int]]: (new_counter_value, awakened_wavefront_id_or_None)
        """
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


def ref_ds_gws_sema_p(subsystem: GWSSubsystem, m0_val: int, offset: int, wave_id: int) -> Tuple[bool, int, bool]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_p."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.wait_p(rid, wave_id)


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
    DS_GWS_SEMA_P = "ds_gws_sema_p"
    DS_GWS_SEMA_V = "ds_gws_sema_v"
    DS_GWS_INIT = "ds_gws_init"
    DS_GWS_BARRIER = "ds_gws_barrier"


@dataclass(frozen=True)
class Instruction:
    """AST representation of RDNA2 ds_gws_sema_p instruction."""
    opcode: Opcode = Opcode.DS_GWS_SEMA_P
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

        offset = 0
        gds = False
        vdata = None

        for tok in tokens[1:]:
            if tok == "gds":
                gds = True
            elif tok.startswith("offset:"):
                raw_off = tok.split(":", 1)[1]
                offset = int(raw_off, 0)
            elif tok.startswith("v") and tok[1:].isdigit():
                vdata = Register(int(tok[1:]), RegKind.VGPR)
            else:
                raise ValueError(f"Unexpected token in instruction: {tok}")

        if not gds:
            raise ValueError("Instruction missing required 'gds' modifier")

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
class SemaphoreAcquireExpr:
    """Symbolic typed expression for semaphore acquire condition."""
    resource_id: int

    def can_acquire(self, counter: int) -> bool:
        return (counter & MASK32) > 0

    def next_counter(self, counter: int) -> int:
        c = counter & MASK32
        return (c - 1) & MASK32 if c > 0 else 0


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
    """Deterministic scalar instruction interpreter for GWS instructions."""

    def execute(self, state: ScalarThreadState, inst: Instruction) -> Tuple[bool, int, bool]:
        """Execute a single GWS instruction deterministically on scalar state.

        Returns:
            Tuple[bool, int, bool]: (acquired, new_counter_val, blocked)
        """
        if state.status == ThreadExecutionStatus.BLOCKED:
            raise RuntimeError(f"Cannot execute instruction on suspended wavefront {state.wave_id}")

        if inst.opcode != Opcode.DS_GWS_SEMA_P:
            raise ValueError(f"Unsupported opcode in ScalarGWSInterpreter: {inst.opcode}")

        # Check if lane 0 is active in exec_mask
        if (state.exec_mask & 1) == 0:
            rid = ref_gws_resource_id(state.m0, inst.offset)
            current_val = state.gws.resources[rid].value
            return False, current_val, False

        rid = ref_gws_resource_id(state.m0, inst.offset)
        acquired, val, blocked = state.gws.wait_p(rid, state.wave_id)

        if acquired:
            state.status = ThreadExecutionStatus.RUNNING
            state.scc = 1 if val > 0 else 0
            return True, val, False
        else:
            state.status = ThreadExecutionStatus.BLOCKED
            state.scc = 0
            return False, 0, True


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

    def execute(self, wave: SIMDWaveState, inst: Instruction) -> Tuple[bool, int, bool]:
        """Execute a GWS instruction on a full wavefront under EXEC mask.

        Returns:
            Tuple[bool, int, bool]: (acquired, new_counter_val, blocked)
        """
        if wave.status == ThreadExecutionStatus.BLOCKED:
            raise RuntimeError(f"Cannot execute instruction on suspended wavefront {wave.wave_id}")

        if inst.opcode != Opcode.DS_GWS_SEMA_P:
            raise ValueError(f"Unsupported opcode in SIMDGWSInterpreter: {inst.opcode}")

        active_lane = wave.first_active_lane()
        if active_lane is None:
            # EXEC == 0: whole-wave instruction is suppressed
            rid = ref_gws_resource_id(wave.m0, inst.offset)
            return False, wave.gws.resources[rid].value, False

        rid = ref_gws_resource_id(wave.m0, inst.offset)
        acquired, val, blocked = wave.gws.wait_p(rid, wave.wave_id)

        if acquired:
            wave.status = ThreadExecutionStatus.RUNNING
            return True, val, False
        else:
            wave.status = ThreadExecutionStatus.BLOCKED
            return False, 0, True


# ============================================================================
# Commit 6: Condition codes, status flags, and carry/borrow generation
# ============================================================================


@dataclass(frozen=True)
class GWSStatusFlags:
    """Architectural condition codes and hardware status flags for GWS operations."""
    zero_flag: bool = False
    negative_flag: bool = False
    borrow_flag: bool = False
    halt_flag: bool = False
    gws_pending: bool = False

    @classmethod
    def compute(cls, acquired: bool, counter_val: int, blocked: bool) -> GWSStatusFlags:
        """Derive hardware condition codes from GWS operation result."""
        # Zero flag is set when counter was decremented to 0
        zero = acquired and (counter_val == 0)
        # Negative flag is always False: GWS counters are unsigned and never wrap negative
        negative = False
        # Borrow flag: prevented because counter never underflows below 0
        borrow = False
        # Halt and pending flags are set when wavefront blocks on zero counter
        halt = blocked
        pending = blocked
        return cls(
            zero_flag=zero,
            negative_flag=negative,
            borrow_flag=borrow,
            halt_flag=halt,
            gws_pending=pending,
        )


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_acquire_decrements_counter(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=1)
        acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=10)
        self.assertTrue(acquired)
        self.assertEqual(val, 0)
        self.assertFalse(blocked)
        self.assertEqual(self.gws.resources[rid].value, 0)
        self.assertEqual(self.gws.resources[rid].acquire_count, 1)

    def test_consecutive_acquires_until_empty(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=10)
        for expected_val in range(9, -1, -1):
            acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=1)
            self.assertTrue(acquired)
            self.assertEqual(val, expected_val)
            self.assertFalse(blocked)
        self.assertEqual(self.gws.resources[rid].value, 0)
        self.assertEqual(self.gws.resources[rid].acquire_count, 10)

    def test_acquire_on_zero_counter_blocks_wavefront(self) -> None:
        rid = 12
        self.gws.init_resource(rid, init_val=0)
        acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=42)
        self.assertFalse(acquired)
        self.assertEqual(val, 0)
        self.assertTrue(blocked)
        self.assertEqual(self.gws.resources[rid].wait_queue, [42])
        self.assertEqual(self.gws.resources[rid].block_count, 1)

    def test_fifo_queue_ordering_for_multiple_waiting_wavefronts(self) -> None:
        rid = 15
        self.gws.init_resource(rid, init_val=0)
        waiting_waves = [101, 102, 103, 104, 105]
        for w in waiting_waves:
            acquired, _, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=w)
            self.assertFalse(acquired)
            self.assertTrue(blocked)
        self.assertEqual(self.gws.resources[rid].wait_queue, waiting_waves)

        # Signals should wake wavefronts in exact FIFO order
        for expected_w in waiting_waves:
            _, awakened = self.gws.signal_v(rid)
            self.assertEqual(awakened, expected_w)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_paired_signal_v_and_wait_p(self) -> None:
        rid = 20
        # Start at 0, wave 99 blocks
        self.gws.wait_p(rid, wave_id=99)
        self.assertEqual(self.gws.resources[rid].wait_queue, [99])

        # Signal wakes wave 99, counter stays 0
        new_val, awakened = self.gws.signal_v(rid)
        self.assertEqual(awakened, 99)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.gws.resources[rid].wait_queue, [])

        # Another signal increments counter to 1
        new_val, awakened = self.gws.signal_v(rid)
        self.assertIsNone(awakened)
        self.assertEqual(new_val, 1)

        # Subsequent wait_p acquires immediately
        acquired, val, blocked = ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=100)
        self.assertTrue(acquired)
        self.assertEqual(val, 0)
        self.assertFalse(blocked)

    def test_resource_id_modulo_64_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)
        self.assertEqual(ref_gws_resource_id(m0_val=0, offset=0x10005), 5)

    def test_independent_gws_resource_isolation(self) -> None:
        self.gws.init_resource(1, init_val=5)
        self.gws.init_resource(2, init_val=10)

        ref_ds_gws_sema_p(self.gws, m0_val=1, offset=0, wave_id=1)
        self.assertEqual(self.gws.resources[1].value, 4)
        self.assertEqual(self.gws.resources[2].value, 10)

        ref_ds_gws_sema_p(self.gws, m0_val=2, offset=0, wave_id=2)
        self.assertEqual(self.gws.resources[1].value, 4)
        self.assertEqual(self.gws.resources[2].value, 9)

    def test_barrier_mode_rejection(self) -> None:
        rid = 8
        self.gws.init_resource(rid, init_val=0, res_type=GWSResourceType.BARRIER)
        with self.assertRaises(TypeError):
            ref_ds_gws_sema_p(self.gws, m0_val=rid, offset=0, wave_id=1)


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions for ds_gws_sema_p."""

    def test_register_construction_and_formatting(self) -> None:
        v0 = Register(0, RegKind.VGPR)
        v255 = Register(255, RegKind.VGPR)
        s15 = Register(15, RegKind.SGPR)
        m0 = Register(0, RegKind.M0)
        exec_reg = Register(0, RegKind.EXEC)
        vcc_reg = Register(0, RegKind.VCC)
        scc_reg = Register(0, RegKind.SCC)
        status_reg = Register(0, RegKind.STATUS)

        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        self.assertEqual(str(s15), "s15")
        self.assertEqual(str(m0), "m0")
        self.assertEqual(str(exec_reg), "exec")
        self.assertEqual(str(vcc_reg), "vcc")
        self.assertEqual(str(scc_reg), "scc")
        self.assertEqual(str(status_reg), "status")

    def test_register_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Register(-1, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(256, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(1, RegKind.M0)
        with self.assertRaises(ValueError):
            Register(2, RegKind.EXEC)

    def test_instruction_disassembly(self) -> None:
        inst1 = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)
        self.assertEqual(inst1.disasm(), "ds_gws_sema_p offset:0 gds")

        inst2 = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=15, gds=True)
        self.assertEqual(inst2.disasm(), "ds_gws_sema_p offset:15 gds")

        inst3 = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=4, gds=True, vdata=Register(2, RegKind.VGPR))
        self.assertEqual(inst3.disasm(), "ds_gws_sema_p v2 offset:4 gds")

    def test_instruction_assembly_parser(self) -> None:
        inst1 = Instruction.parse("ds_gws_sema_p offset:0 gds")
        self.assertEqual(inst1.opcode, Opcode.DS_GWS_SEMA_P)
        self.assertEqual(inst1.offset, 0)
        self.assertTrue(inst1.gds)
        self.assertIsNone(inst1.vdata)

        inst2 = Instruction.parse("ds_gws_sema_p offset:42 gds")
        self.assertEqual(inst2.offset, 42)

        inst3 = Instruction.parse("ds_gws_sema_p v1 offset:8 gds")
        self.assertIsNotNone(inst3.vdata)
        self.assertEqual(inst3.vdata.index, 1)
        self.assertEqual(inst3.offset, 8)

    def test_instruction_parser_rejections(self) -> None:
        with self.assertRaises(ValueError):
            Instruction.parse("ds_gws_sema_p offset:0")  # missing gds
        with self.assertRaises(ValueError):
            Instruction.parse("v_add_f32 v0, v1, v2")  # wrong opcode
        with self.assertRaises(ValueError):
            Instruction.parse("ds_gws_sema_p offset:70000 gds")  # out of bounds offset

    def test_gws_resource_expr_evaluation(self) -> None:
        expr = GWSResourceExpr(offset=10)
        self.assertEqual(expr.evaluate(m0_val=2), 12)
        self.assertEqual(expr.evaluate(m0_val=60), 6)  # (60 + 10) % 64 = 6
        self.assertEqual(expr.evaluate(m0_val=0), 10)

    def test_semaphore_acquire_expr(self) -> None:
        expr = SemaphoreAcquireExpr(resource_id=3)
        self.assertTrue(expr.can_acquire(5))
        self.assertEqual(expr.next_counter(5), 4)

        self.assertTrue(expr.can_acquire(1))
        self.assertEqual(expr.next_counter(1), 0)

        self.assertFalse(expr.can_acquire(0))
        self.assertEqual(expr.next_counter(0), 0)


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter for ds_gws_sema_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.interp = ScalarGWSInterpreter()

    def test_scalar_acquire_success(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=2)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 1)
        self.assertFalse(blocked)
        self.assertEqual(state.status, ThreadExecutionStatus.RUNNING)
        self.assertEqual(state.scc, 1)  # remaining counter > 0

        # Second acquire decrements to 0
        acquired2, val2, blocked2 = self.interp.execute(state, inst)
        self.assertTrue(acquired2)
        self.assertEqual(val2, 0)
        self.assertFalse(blocked2)
        self.assertEqual(state.status, ThreadExecutionStatus.RUNNING)
        self.assertEqual(state.scc, 0)  # remaining counter == 0

    def test_scalar_acquire_blocks_when_zero(self) -> None:
        rid = 6
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=7, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertFalse(acquired)
        self.assertEqual(val, 0)
        self.assertTrue(blocked)
        self.assertEqual(state.status, ThreadExecutionStatus.BLOCKED)
        self.assertEqual(state.scc, 0)
        self.assertEqual(self.gws.resources[rid].wait_queue, [7])

    def test_blocked_thread_rejection(self) -> None:
        rid = 10
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=5, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        self.interp.execute(state, inst)
        self.assertEqual(state.status, ThreadExecutionStatus.BLOCKED)

        with self.assertRaises(RuntimeError):
            self.interp.execute(state, inst)

    def test_waking_blocked_thread(self) -> None:
        rid = 11
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=22, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        self.interp.execute(state, inst)
        self.assertEqual(state.status, ThreadExecutionStatus.BLOCKED)

        # External signal_v wakes wavefront 22
        val, awakened = self.gws.signal_v(rid)
        self.assertEqual(awakened, 22)
        # Restore status to RUNNING upon wakeup
        state.status = ThreadExecutionStatus.RUNNING
        self.assertEqual(state.status, ThreadExecutionStatus.RUNNING)

    def test_inactive_exec_mask_is_noop(self) -> None:
        rid = 14
        self.gws.init_resource(rid, init_val=5)
        state = ScalarThreadState(wave_id=1, m0=rid, exec_mask=0, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertFalse(acquired)
        self.assertEqual(val, 5)
        self.assertFalse(blocked)
        self.assertEqual(self.gws.resources[rid].value, 5)
        self.assertEqual(state.status, ThreadExecutionStatus.RUNNING)

    def test_dynamic_m0_offset_resolution(self) -> None:
        self.gws.init_resource(12, init_val=3)
        state = ScalarThreadState(wave_id=1, m0=10, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=2, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 2)
        self.assertEqual(self.gws.resources[12].value, 2)


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for ds_gws_sema_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.interp = SIMDGWSInterpreter()

    def test_wave32_full_exec_single_decrement(self) -> None:
        rid = 1
        self.gws.init_resource(rid, init_val=10)
        wave = SIMDWaveState(wave_id=10, wave_size=32, exec_mask=0xFFFF_FFFF, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertTrue(acquired)
        # Must decrement by exactly 1 for the whole wavefront, not 32!
        self.assertEqual(val, 9)
        self.assertEqual(self.gws.resources[rid].value, 9)
        self.assertEqual(self.gws.resources[rid].acquire_count, 1)

    def test_wave32_divergent_mask_single_lane_active(self) -> None:
        rid = 2
        self.gws.init_resource(rid, init_val=5)
        # Only lane 19 is active
        wave = SIMDWaveState(wave_id=11, wave_size=32, exec_mask=(1 << 19), m0=rid, gws=self.gws)
        self.assertEqual(wave.first_active_lane(), 19)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 4)
        self.assertEqual(self.gws.resources[rid].value, 4)

    def test_wave32_alternating_lanes_mask(self) -> None:
        rid = 3
        self.gws.init_resource(rid, init_val=2)
        wave = SIMDWaveState(wave_id=12, wave_size=32, exec_mask=0x5555_5555, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 1)

    def test_wave32_exec_zero_suppression(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=7)
        wave = SIMDWaveState(wave_id=13, wave_size=32, exec_mask=0, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertFalse(acquired)
        self.assertEqual(val, 7)
        self.assertFalse(blocked)
        self.assertEqual(self.gws.resources[rid].value, 7)
        self.assertEqual(wave.status, ThreadExecutionStatus.RUNNING)

    def test_wave64_full_exec_single_decrement(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=15)
        wave = SIMDWaveState(wave_id=20, wave_size=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 14)
        self.assertEqual(self.gws.resources[rid].value, 14)

    def test_wave64_high_lanes_only(self) -> None:
        rid = 6
        self.gws.init_resource(rid, init_val=3)
        # Upper 32 lanes active only
        wave = SIMDWaveState(wave_id=21, wave_size=64, exec_mask=0xFFFF_FFFF_0000_0000, m0=rid, gws=self.gws)
        self.assertEqual(wave.first_active_lane(), 32)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 2)

    def test_wave64_exec_zero_suppression(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=8)
        wave = SIMDWaveState(wave_id=22, wave_size=64, exec_mask=0, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertFalse(acquired)
        self.assertEqual(val, 8)
        self.assertFalse(blocked)
        self.assertEqual(wave.status, ThreadExecutionStatus.RUNNING)

    def test_wave_blocks_when_zero_under_exec(self) -> None:
        rid = 8
        self.gws.init_resource(rid, init_val=0)
        wave = SIMDWaveState(wave_id=99, wave_size=32, exec_mask=0x0000_0001, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(wave, inst)
        self.assertFalse(acquired)
        self.assertEqual(val, 0)
        self.assertTrue(blocked)
        self.assertEqual(wave.status, ThreadExecutionStatus.BLOCKED)
        self.assertEqual(self.gws.resources[rid].wait_queue, [99])


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation for ds_gws_sema_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.scalar_interp = ScalarGWSInterpreter()
        self.simd_interp = SIMDGWSInterpreter()

    def test_m0_preservation_after_successful_acquire(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=4)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        self.scalar_interp.execute(state, inst)
        self.assertEqual(state.m0, rid)

    def test_m0_preservation_after_blocked_acquire(self) -> None:
        rid = 9
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        self.scalar_interp.execute(state, inst)
        self.assertEqual(state.m0, rid)

    def test_all_sgprs_preserved_in_place(self) -> None:
        rid = 3
        self.gws.init_resource(rid, init_val=5)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        # Populate various SGPRs with distinct bit patterns
        for s in range(106):
            state.set_sgpr(s, (0xCAFE0000 + s * 17) & MASK32)
        saved_sgprs = dict(state.sgprs)

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)
        self.scalar_interp.execute(state, inst)

        self.assertEqual(state.sgprs, saved_sgprs)

    def test_all_lane_vgprs_preserved_in_simd(self) -> None:
        rid = 10
        self.gws.init_resource(rid, init_val=2)
        wave = SIMDWaveState(wave_id=2, wave_size=32, m0=rid, gws=self.gws)
        # Populate VGPRs across all 32 lanes
        for lane in range(32):
            for v in range(0, 256, 32):
                wave.set_vgpr(lane, v, (lane << 16) | v)
        saved_vgprs = [dict(lane_dict) for lane_dict in wave.lane_vgprs]

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)
        self.simd_interp.execute(wave, inst)

        self.assertEqual(wave.lane_vgprs, saved_vgprs)

    def test_aliased_m0_and_sgpr_operand_integrity(self) -> None:
        rid = 14
        self.gws.init_resource(rid, init_val=1)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_sgpr(0, rid)
        state.set_sgpr(1, rid)

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)
        self.scalar_interp.execute(state, inst)

        self.assertEqual(state.m0, rid)
        self.assertEqual(state.get_sgpr(0), rid)
        self.assertEqual(state.get_sgpr(1), rid)

    def test_vdata_operand_preservation(self) -> None:
        rid = 18
        self.gws.init_resource(rid, init_val=3)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_vgpr(5, 0x12345678)

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True, vdata=Register(5, RegKind.VGPR))
        self.scalar_interp.execute(state, inst)

        self.assertEqual(state.get_vgpr(5), 0x12345678)

    def test_consecutive_pipeline_operand_preservation(self) -> None:
        rid = 22
        self.gws.init_resource(rid, init_val=3)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_sgpr(10, 0xDEADBEEF)
        state.set_vgpr(20, 0xBEEFBEEF)

        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)
        for _ in range(3):
            self.scalar_interp.execute(state, inst)
            self.assertEqual(state.m0, rid)
            self.assertEqual(state.get_sgpr(10), 0xDEADBEEF)
            self.assertEqual(state.get_vgpr(20), 0xBEEFBEEF)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation for ds_gws_sema_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.interp = ScalarGWSInterpreter()

    def test_zero_flag_set_on_counter_exhaustion(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=1)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        flags = GWSStatusFlags.compute(acquired, val, blocked)

        self.assertTrue(flags.zero_flag)
        self.assertFalse(flags.negative_flag)
        self.assertFalse(flags.borrow_flag)
        self.assertFalse(flags.halt_flag)
        self.assertFalse(flags.gws_pending)
        self.assertEqual(state.scc, 0)

    def test_zero_flag_clear_when_counter_remains_positive(self) -> None:
        rid = 6
        self.gws.init_resource(rid, init_val=5)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        flags = GWSStatusFlags.compute(acquired, val, blocked)

        self.assertFalse(flags.zero_flag)
        self.assertFalse(flags.negative_flag)
        self.assertFalse(flags.borrow_flag)
        self.assertFalse(flags.halt_flag)
        self.assertEqual(state.scc, 1)

    def test_borrow_prevention_on_zero_counter(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        flags = GWSStatusFlags.compute(acquired, val, blocked)

        # Counter must NOT underflow to 0xFFFFFFFF
        self.assertEqual(val, 0)
        self.assertEqual(self.gws.resources[rid].value, 0)
        self.assertFalse(flags.borrow_flag)
        self.assertTrue(flags.halt_flag)
        self.assertTrue(flags.gws_pending)
        self.assertEqual(state.status, ThreadExecutionStatus.BLOCKED)

    def test_status_flag_transition_on_signal_v_wakeup(self) -> None:
        rid = 8
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=44, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        # Thread blocks
        acquired, val, blocked = self.interp.execute(state, inst)
        flags_blocked = GWSStatusFlags.compute(acquired, val, blocked)
        self.assertTrue(flags_blocked.halt_flag)
        self.assertTrue(flags_blocked.gws_pending)

        # Paired signal_v awakens wave 44
        _, awakened = self.gws.signal_v(rid)
        self.assertEqual(awakened, 44)

        # Wakeup clears halt and pending
        state.status = ThreadExecutionStatus.RUNNING
        flags_resumed = GWSStatusFlags.compute(acquired=True, counter_val=0, blocked=False)
        self.assertFalse(flags_resumed.halt_flag)
        self.assertFalse(flags_resumed.gws_pending)

    def test_scc_tracking_across_depletion_sequence(self) -> None:
        rid = 9
        self.gws.init_resource(rid, init_val=2)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        # 2 -> 1: scc=1 (still > 0)
        self.interp.execute(state, inst)
        self.assertEqual(state.scc, 1)

        # 1 -> 0: scc=0 (hit 0)
        self.interp.execute(state, inst)
        self.assertEqual(state.scc, 0)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling for ds_gws_sema_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.interp = ScalarGWSInterpreter()

    def test_positive_zero_float_representation_blocks(self) -> None:
        # Float +0.0 is bitwise 0x0000_0000: counter is 0 -> blocks
        pos_zero_bits = struct.unpack(">I", struct.pack(">f", 0.0))[0]
        self.assertEqual(pos_zero_bits, 0x0000_0000)

        rid = 1
        self.gws.init_resource(rid, init_val=pos_zero_bits)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertFalse(acquired)
        self.assertTrue(blocked)
        self.assertEqual(val, 0)

    def test_negative_zero_float_representation_does_not_block(self) -> None:
        # Float -0.0 is bitwise 0x8000_0000: as unsigned integer it is 2^31 > 0 -> acquires!
        neg_zero_bits = struct.unpack(">I", struct.pack(">f", -0.0))[0]
        self.assertEqual(neg_zero_bits, 0x8000_0000)

        rid = 2
        self.gws.init_resource(rid, init_val=neg_zero_bits)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        self.assertFalse(blocked)
        self.assertEqual(val, 0x7FFF_FFFF)

    def test_min_subnormal_float_decrements_to_zero(self) -> None:
        # 0x0000_0001: smallest positive subnormal float (approx 1.4e-45)
        min_subnormal = 0x0000_0001
        rid = 3
        self.gws.init_resource(rid, init_val=min_subnormal)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 0x0000_0000)
        self.assertEqual(state.scc, 0)

    def test_max_subnormal_float_preserved_without_ftz_flush(self) -> None:
        # 0x007F_FFFF: largest positive subnormal float
        max_subnormal = 0x007F_FFFF
        rid = 4
        self.gws.init_resource(rid, init_val=max_subnormal)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        # Decrements by 1 without flush-to-zero (FTZ)
        self.assertEqual(val, 0x007F_FFFE)

    def test_positive_infinity_decrements_to_max_finite_float(self) -> None:
        # Float +inf is bitwise 0x7F80_0000. Decrementing by 1 yields 0x7F7F_FFFF (FLT_MAX)
        pos_inf_bits = struct.unpack(">I", struct.pack(">f", float("inf")))[0]
        self.assertEqual(pos_inf_bits, 0x7F80_0000)

        rid = 5
        self.gws.init_resource(rid, init_val=pos_inf_bits)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 0x7F7F_FFFF)

        # Confirm 0x7F7F_FFFF unpacks to a finite float
        unpacked_float = struct.unpack(">f", struct.pack(">I", val))[0]
        self.assertTrue(math.isfinite(unpacked_float))
        self.assertGreater(unpacked_float, 3.4e38)

    def test_negative_infinity_decrements_integer_representation(self) -> None:
        neg_inf_bits = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]
        self.assertEqual(neg_inf_bits, 0xFF80_0000)

        rid = 6
        self.gws.init_resource(rid, init_val=neg_inf_bits)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 0xFF7F_FFFF)

    def test_quiet_nan_preserved_without_canonicalization(self) -> None:
        # Float qNaN: 0x7FC0_0000
        qnan_bits = 0x7FC0_0000
        rid = 7
        self.gws.init_resource(rid, init_val=qnan_bits)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        # Strict bit-exact arithmetic without NaN canonicalization
        self.assertEqual(val, 0x7FBF_FFFF)

    def test_signaling_nan_decrements_to_infinity(self) -> None:
        # Float sNaN: 0x7F80_0001
        snan_bits = 0x7F80_0001
        rid = 8
        self.gws.init_resource(rid, init_val=snan_bits)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

        acquired, val, blocked = self.interp.execute(state, inst)
        self.assertTrue(acquired)
        self.assertEqual(val, 0x7F80_0000)  # exactly bit pattern of +inf

    def test_m0_populated_with_float_bit_patterns(self) -> None:
        # If M0 holds a float bit pattern, only low 6 bits are used for resource ID
        float_patterns = [
            0x7FC0_0005,  # NaN with low 6 bits = 5
            0x7F80_000A,  # Inf with low 6 bits = 10
            0x8000_0010,  # -0.0 with low 6 bits = 16
        ]
        for pattern in float_patterns:
            expected_rid = pattern & GWS_RESOURCE_ID_MASK
            self.gws.init_resource(expected_rid, init_val=3)
            state = ScalarThreadState(wave_id=1, m0=pattern, gws=self.gws)
            inst = Instruction(opcode=Opcode.DS_GWS_SEMA_P, offset=0, gds=True)

            acquired, val, blocked = self.interp.execute(state, inst)
            self.assertTrue(acquired)
            self.assertEqual(val, 2)
            self.assertEqual(self.gws.resources[expected_rid].value, 2)


if __name__ == "__main__":
    unittest.main()
