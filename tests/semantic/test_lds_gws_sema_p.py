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


if __name__ == "__main__":
    unittest.main()
