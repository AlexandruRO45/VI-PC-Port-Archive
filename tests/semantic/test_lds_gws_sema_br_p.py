from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 ds_gws_sema_br_p.

Models AMD RDNA2 / GFX10 Global Wavefront Sync (GWS) hardware semaphore
branch-probe operation (non-blocking try-acquire / probe-and-branch):
  - ds_gws_sema_br_p: Probes the targeted GWS semaphore resource counter.
    If counter > 0 (or >= requested permits), decrements counter by requested
    amount (default 1) and signals successful probe (branch taken / SCC=1).
    If counter == 0 (or < requested permits), the probe fails (branch not taken / SCC=0).
    CRITICALLY: unlike blocking ds_gws_sema_p, ds_gws_sema_br_p NEVER blocks
    or enqueues the wavefront! It is a non-blocking speculative probe.

RDNA2 GWS Architectural Rules:
  1. GWS Resource Addressing:
     - 64 hardware resource slots (0..63) in GDS space.
     - Effective target resource ID: (m0 + (offset16 & 0xFFFF)) & 0x3F.
  2. Semaphore Branch Probe Semantics:
     - Probe Condition: Checks if counter >= required_permits (default 1).
     - Success Path: Counter is atomically decremented: counter -= required_permits.
       Probe returns (success=True, new_counter=counter, branch_taken=True).
     - Failure Path: Counter is unchanged. Probe returns (success=False, new_counter=counter, branch_taken=False).
     - Non-Blocking Guarantee: The calling wavefront never halts, suspends, or sleeps.
  3. Resource Isolation & Mode Enforcement:
     - Each resource slot is independently configured (UNINITIALIZED, SEMAPHORE, BARRIER).
     - Executing ds_gws_sema_br_p on a BARRIER resource triggers an architectural type fault.

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
    probe_success_count: int = 0
    probe_fail_count: int = 0

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
        res.probe_success_count = 0
        res.probe_fail_count = 0

    def probe_p(self, res_id: int, permits: int = 1) -> Tuple[bool, int, bool]:
        """Executes a semaphore branch probe (try-acquire) operation.

        Returns:
            Tuple[bool, int, bool]: (probe_success, new_counter_value, branch_taken)
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_br_p")
        res.res_type = GWSResourceType.SEMAPHORE

        req_permits = max(1, permits & MASK32)
        if res.value >= req_permits:
            res.value = (res.value - req_permits) & MASK32
            res.probe_success_count += 1
            return True, res.value, True
        else:
            res.probe_fail_count += 1
            return False, res.value, False

    def wait_p(self, res_id: int, wave_id: int) -> Tuple[bool, int, bool]:
        """Executes a standard blocking semaphore P operation."""
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
        """Executes a semaphore V (signal/release) operation."""
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


def ref_ds_gws_sema_br_p(subsystem: GWSSubsystem, m0_val: int, offset: int, permits: int = 1) -> Tuple[bool, int, bool]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_br_p."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.probe_p(rid, permits=permits)


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
    DS_GWS_SEMA_BR_P = "ds_gws_sema_br_p"
    DS_GWS_SEMA_BR = "ds_gws_sema_br"
    DS_GWS_SEMA_P = "ds_gws_sema_p"
    DS_GWS_SEMA_V = "ds_gws_sema_v"
    DS_GWS_INIT = "ds_gws_init"
    DS_GWS_BARRIER = "ds_gws_barrier"


@dataclass(frozen=True)
class Instruction:
    """AST representation of RDNA2 ds_gws_sema_br_p instruction."""
    opcode: Opcode = Opcode.DS_GWS_SEMA_BR_P
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
class SemaphoreBranchProbeExpr:
    """Symbolic typed expression for semaphore branch probe condition."""
    resource_id: int

    def can_probe(self, counter: int, required_permits: int = 1) -> bool:
        c = counter & MASK32
        req = max(1, required_permits & MASK32)
        return c >= req

    def next_counter(self, counter: int, required_permits: int = 1) -> int:
        c = counter & MASK32
        req = max(1, required_permits & MASK32)
        return (c - req) & MASK32 if c >= req else c

    def branch_taken(self, counter: int, required_permits: int = 1) -> bool:
        return self.can_probe(counter, required_permits)


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
    """Deterministic scalar instruction interpreter for GWS branch probe instructions."""

    def execute(
        self,
        state: ScalarThreadState,
        inst: Instruction,
        branch_pc: Optional[int] = None,
    ) -> Tuple[bool, int, bool]:
        """Execute a single GWS branch probe instruction deterministically.

        Returns:
            Tuple[bool, int, bool]: (probe_success, new_counter_val, branch_taken)
        """
        if state.status == ThreadExecutionStatus.BLOCKED:
            raise RuntimeError(f"Cannot execute instruction on suspended wavefront {state.wave_id}")

        if inst.opcode != Opcode.DS_GWS_SEMA_BR_P:
            raise ValueError(f"Unsupported opcode in ScalarGWSInterpreter: {inst.opcode}")

        # Inactive lane in EXEC mask: no-op
        if (state.exec_mask & 1) == 0:
            rid = ref_gws_resource_id(state.m0, inst.offset)
            current_val = state.gws.resources[rid].value
            return False, current_val, False

        rid = ref_gws_resource_id(state.m0, inst.offset)

        # Determine requested permits from vdata or default to 1
        permits = 1
        if inst.vdata is not None:
            permits = max(1, state.get_vgpr(inst.vdata.index))

        success, val, branch_taken = state.gws.probe_p(rid, permits=permits)

        if success:
            state.scc = 1
            state.branch_target = branch_pc
            state.status = ThreadExecutionStatus.RUNNING
            return True, val, True
        else:
            state.scc = 0
            state.branch_target = None
            state.status = ThreadExecutionStatus.RUNNING
            return False, val, False


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_br_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_probe_success(self) -> None:
        rid = 3
        self.gws.init_resource(rid, init_val=1)
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(success)
        self.assertEqual(val, 0)
        self.assertTrue(branch)
        self.assertEqual(self.gws.resources[rid].value, 0)
        self.assertEqual(self.gws.resources[rid].probe_success_count, 1)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_single_probe_fail_when_zero_never_blocks(self) -> None:
        rid = 5
        self.gws.init_resource(rid, init_val=0)
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertFalse(success)
        self.assertEqual(val, 0)
        self.assertFalse(branch)
        # CRITICAL: wait_queue must be empty! Probe never suspends the calling wave
        self.assertEqual(self.gws.resources[rid].wait_queue, [])
        self.assertEqual(self.gws.resources[rid].probe_fail_count, 1)

    def test_multi_permit_probe_success(self) -> None:
        rid = 8
        self.gws.init_resource(rid, init_val=10)
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=4)
        self.assertTrue(success)
        self.assertEqual(val, 6)
        self.assertTrue(branch)
        self.assertEqual(self.gws.resources[rid].value, 6)

    def test_multi_permit_probe_fail_insufficient(self) -> None:
        rid = 9
        self.gws.init_resource(rid, init_val=3)
        # Requesting 4 permits when only 3 available
        success, val, branch = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=4)
        self.assertFalse(success)
        self.assertEqual(val, 3)  # counter is NOT decremented
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[rid].value, 3)

    def test_consecutive_probes_until_exhaustion(self) -> None:
        rid = 12
        self.gws.init_resource(rid, init_val=3)

        # Probe 1: succeeds (3 -> 2)
        s1, v1, b1 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s1)
        self.assertEqual(v1, 2)
        self.assertTrue(b1)

        # Probe 2: succeeds (2 -> 1)
        s2, v2, b2 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s2)
        self.assertEqual(v2, 1)
        self.assertTrue(b2)

        # Probe 3: succeeds (1 -> 0)
        s3, v3, b3 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s3)
        self.assertEqual(v3, 0)
        self.assertTrue(b3)

        # Probe 4: fails (0 remaining)
        s4, v4, b4 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertFalse(s4)
        self.assertEqual(v4, 0)
        self.assertFalse(b4)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_interleaved_probe_and_signal_v(self) -> None:
        rid = 16
        self.gws.init_resource(rid, init_val=0)

        # Initially fails
        s1, v1, b1 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertFalse(s1)

        # Signal V adds 1 permit
        self.gws.signal_v(rid)
        self.assertEqual(self.gws.resources[rid].value, 1)

        # Now probe succeeds
        s2, v2, b2 = ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)
        self.assertTrue(s2)
        self.assertEqual(v2, 0)
        self.assertTrue(b2)

    def test_resource_id_modulo_64_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)
        self.assertEqual(ref_gws_resource_id(m0_val=0, offset=0x10005), 5)

    def test_resource_isolation(self) -> None:
        self.gws.init_resource(2, init_val=5)
        self.gws.init_resource(3, init_val=10)

        ref_ds_gws_sema_br_p(self.gws, m0_val=2, offset=0, permits=1)
        self.assertEqual(self.gws.resources[2].value, 4)
        self.assertEqual(self.gws.resources[3].value, 10)

        ref_ds_gws_sema_br_p(self.gws, m0_val=3, offset=0, permits=2)
        self.assertEqual(self.gws.resources[2].value, 4)
        self.assertEqual(self.gws.resources[3].value, 8)

    def test_barrier_mode_rejection(self) -> None:
        rid = 4
        self.gws.init_resource(rid, init_val=5, res_type=GWSResourceType.BARRIER)
        with self.assertRaises(TypeError):
            ref_ds_gws_sema_br_p(self.gws, m0_val=rid, offset=0, permits=1)


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions for ds_gws_sema_br_p."""

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
        inst1 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=0, gds=True)
        self.assertEqual(inst1.disasm(), "ds_gws_sema_br_p offset:0 gds")

        inst2 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=15, gds=True)
        self.assertEqual(inst2.disasm(), "ds_gws_sema_br_p offset:15 gds")

        inst3 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=4, gds=True, vdata=Register(2, RegKind.VGPR))
        self.assertEqual(inst3.disasm(), "ds_gws_sema_br_p v2 offset:4 gds")

    def test_instruction_assembly_parser(self) -> None:
        inst1 = Instruction.parse("ds_gws_sema_br_p offset:0 gds")
        self.assertEqual(inst1.opcode, Opcode.DS_GWS_SEMA_BR_P)
        self.assertEqual(inst1.offset, 0)
        self.assertTrue(inst1.gds)
        self.assertIsNone(inst1.vdata)

        inst2 = Instruction.parse("ds_gws_sema_br_p offset:42 gds")
        self.assertEqual(inst2.offset, 42)

        inst3 = Instruction.parse("ds_gws_sema_br_p v1 offset:8 gds")
        self.assertIsNotNone(inst3.vdata)
        self.assertEqual(inst3.vdata.index, 1)
        self.assertEqual(inst3.offset, 8)

    def test_instruction_parser_rejections(self) -> None:
        with self.assertRaises(ValueError):
            Instruction.parse("ds_gws_sema_br_p offset:0")  # missing gds
        with self.assertRaises(ValueError):
            Instruction.parse("ds_gws_sema_br_p gds offset:0")  # gds not at end
        with self.assertRaises(ValueError):
            Instruction.parse("v_add_f32 v0, v1, v2")  # wrong opcode
        with self.assertRaises(ValueError):
            Instruction.parse("ds_gws_sema_br_p offset:70000 gds")  # out of bounds offset

    def test_gws_resource_expr_evaluation(self) -> None:
        expr = GWSResourceExpr(offset=10)
        self.assertEqual(expr.evaluate(m0_val=2), 12)
        self.assertEqual(expr.evaluate(m0_val=60), 6)  # (60 + 10) % 64 = 6
        self.assertEqual(expr.evaluate(m0_val=0), 10)

    def test_semaphore_branch_probe_expr(self) -> None:
        expr = SemaphoreBranchProbeExpr(resource_id=3)
        # Single permit probe
        self.assertTrue(expr.can_probe(5, required_permits=1))
        self.assertEqual(expr.next_counter(5, required_permits=1), 4)
        self.assertTrue(expr.branch_taken(5, required_permits=1))

        # Multi permit probe success
        self.assertTrue(expr.can_probe(5, required_permits=3))
        self.assertEqual(expr.next_counter(5, required_permits=3), 2)

        # Multi permit probe fail
        self.assertFalse(expr.can_probe(2, required_permits=3))
        self.assertEqual(expr.next_counter(2, required_permits=3), 2)
        self.assertFalse(expr.branch_taken(2, required_permits=3))

        # Zero counter
        self.assertFalse(expr.can_probe(0, required_permits=1))
        self.assertEqual(expr.next_counter(0, required_permits=1), 0)


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter for ds_gws_sema_br_p."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()
        self.interp = ScalarGWSInterpreter()

    def test_scalar_probe_success(self) -> None:
        rid = 6
        self.gws.init_resource(rid, init_val=2)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=0, gds=True)

        success, val, branch = self.interp.execute(state, inst, branch_pc=0x1000)
        self.assertTrue(success)
        self.assertEqual(val, 1)
        self.assertTrue(branch)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.branch_target, 0x1000)
        self.assertEqual(state.status, ThreadExecutionStatus.RUNNING)

    def test_scalar_probe_fail_when_zero_never_blocks(self) -> None:
        rid = 7
        self.gws.init_resource(rid, init_val=0)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=0, gds=True)

        success, val, branch = self.interp.execute(state, inst, branch_pc=0x2000)
        self.assertFalse(success)
        self.assertEqual(val, 0)
        self.assertFalse(branch)
        self.assertEqual(state.scc, 0)
        self.assertIsNone(state.branch_target)
        self.assertEqual(state.status, ThreadExecutionStatus.RUNNING)
        self.assertEqual(self.gws.resources[rid].wait_queue, [])

    def test_scalar_probe_with_vdata_permits(self) -> None:
        rid = 10
        self.gws.init_resource(rid, init_val=8)
        state = ScalarThreadState(wave_id=1, m0=rid, gws=self.gws)
        state.set_vgpr(3, 5)  # request 5 permits
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=0, gds=True, vdata=Register(3, RegKind.VGPR))

        success, val, branch = self.interp.execute(state, inst)
        self.assertTrue(success)
        self.assertEqual(val, 3)
        self.assertTrue(branch)
        self.assertEqual(state.scc, 1)

    def test_scalar_probe_inactive_exec(self) -> None:
        rid = 11
        self.gws.init_resource(rid, init_val=5)
        state = ScalarThreadState(wave_id=1, m0=rid, exec_mask=0, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=0, gds=True)

        success, val, branch = self.interp.execute(state, inst)
        self.assertFalse(success)
        self.assertEqual(val, 5)
        self.assertFalse(branch)
        self.assertEqual(self.gws.resources[rid].value, 5)

    def test_dynamic_m0_offset_resolution(self) -> None:
        self.gws.init_resource(15, init_val=4)
        state = ScalarThreadState(wave_id=1, m0=10, gws=self.gws)
        inst = Instruction(opcode=Opcode.DS_GWS_SEMA_BR_P, offset=5, gds=True)

        success, val, branch = self.interp.execute(state, inst)
        self.assertTrue(success)
        self.assertEqual(val, 3)
        self.assertEqual(self.gws.resources[15].value, 3)


if __name__ == "__main__":
    unittest.main()
