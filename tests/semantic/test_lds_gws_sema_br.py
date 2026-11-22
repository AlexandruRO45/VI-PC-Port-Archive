from __future__ import annotations

import enum
import math
import struct
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# AMD RDNA2 / GFX10 GWS (Global Wavefront Sync) Architectural Constants
GWS_NUM_RESOURCES: int = 64  # Hardware semaphore / barrier resource slots (0..63)
GWS_RESOURCE_ID_MASK: int = 0x3F  # 6-bit resource index mask
SEMAPHORE_MAX_VALUE: int = 0xFFFF_FFFF  # 32-bit unsigned wrapping counter
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


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

    def bulk_release_br(self, res_id: int, release_count: int = 1) -> Tuple[int, List[int]]:
        """Executes a semaphore bulk release (broadcast/bulk signal) operation.

        Args:
            res_id: GWS resource index [0..63].
            release_count: Number of permits delivered in the bulk release.

        Returns:
            Tuple of (new_counter_value, list_of_awakened_wavefront_ids).
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_br")
        res.res_type = GWSResourceType.SEMAPHORE

        count = release_count & MASK32
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

        return res.value, awakened

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
    return (base + offset) & GWS_RESOURCE_ID_MASK


def ref_ds_gws_sema_br(subsystem: GWSSubsystem, m0_val: int, offset: int, release_count: int = 1) -> Tuple[int, List[int]]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_br."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.bulk_release_br(rid, release_count=release_count)


# Commit 2: Symbolic AST opcode definitions and typed expressions


class RegKind(enum.Enum):
    """Register kinds in RDNA2 architecture."""
    VGPR = "vgpr"
    SGPR = "sgpr"
    M0 = "m0"
    EXEC = "exec"
    VCC = "vcc"


@dataclass(frozen=True)
class Register:
    """Represents an RDNA2 hardware register."""
    index: int = 0
    kind: RegKind = RegKind.VGPR

    def __post_init__(self) -> None:
        if self.kind in (RegKind.VGPR, RegKind.SGPR):
            if not (0 <= self.index <= 255):
                raise ValueError(f"Register index out of bounds [0, 255]: {self.index}")
        elif self.kind in (RegKind.M0, RegKind.EXEC, RegKind.VCC):
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
    DS_GWS_SEMA_BR = "ds_gws_sema_br"
    DS_GWS_SEMA_V = "ds_gws_sema_v"
    DS_GWS_SEMA_P = "ds_gws_sema_p"
    DS_GWS_INIT = "ds_gws_init"
    DS_GWS_BARRIER = "ds_gws_barrier"


@dataclass(frozen=True)
class Instruction:
    """AST representation of RDNA2 ds_gws_sema_br instruction."""
    opcode: Opcode = Opcode.DS_GWS_SEMA_BR
    vdata: Optional[Register] = None
    offset: int = 0
    gds: bool = True

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= 255):
            raise ValueError(f"GWS immediate offset out of range [0, 255]: {self.offset}")
        if self.vdata is not None and self.vdata.kind != RegKind.VGPR:
            raise TypeError(f"vdata must be a VGPR, got: {self.vdata.kind}")

    def disasm(self) -> str:
        parts = [self.opcode.value]
        if self.vdata is not None:
            parts.append(str(self.vdata))
        if self.offset != 0 or self.vdata is None:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)


@dataclass(frozen=True)
class GWSExpression:
    """Symbolic expression for effective GWS resource identifier."""
    base_reg: Register = field(default_factory=lambda: Register(0, RegKind.M0))
    offset: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= 255):
            raise ValueError(f"Offset out of range [0, 255]: {self.offset}")

    def evaluate(self, m0_val: int) -> int:
        return ref_gws_resource_id(m0_val, self.offset)


# Commit 3: Deterministic scalar instruction interpreter


class ScalarRegisterFile:
    """Models a single wavefront's scalar and vector register view."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 104
        self.m0: int = 0
        self.exec: int = MASK32
        self.vcc: int = 0

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_gws_sema_br."""

    def __init__(self, gws: Optional[GWSSubsystem] = None) -> None:
        self.gws: GWSSubsystem = gws if gws is not None else GWSSubsystem()
        self.rf: ScalarRegisterFile = ScalarRegisterFile()
        self.trace: List[Tuple[str, int, List[int]]] = []

    def step(self, instr: Instruction, wave_id: int = 0) -> Tuple[int, List[int]]:
        """Execute a single ds_gws_sema_br instruction deterministically."""
        if instr.opcode != Opcode.DS_GWS_SEMA_BR:
            raise NotImplementedError(f"Unsupported opcode in scalar interpreter: {instr.opcode}")

        rid = ref_gws_resource_id(self.rf.m0, instr.offset)
        release_count = 1
        if instr.vdata is not None:
            release_count = self.rf.get_vgpr(instr.vdata.index)

        new_val, awakened = self.gws.bulk_release_br(rid, release_count=release_count)
        self.trace.append((instr.disasm(), new_val, awakened))
        return new_val, awakened


class TestGWSSemaBrOracleAndModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_br."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_bulk_release_increments_counter(self) -> None:
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=0, offset=0, release_count=1)
        self.assertEqual(val, 1)
        self.assertEqual(awakened, [])
        self.assertEqual(self.gws.resources[0].value, 1)

    def test_multi_count_bulk_release(self) -> None:
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=2, offset=3, release_count=16)
        self.assertEqual(val, 16)
        self.assertEqual(awakened, [])
        self.assertEqual(self.gws.resources[5].value, 16)
        self.assertEqual(self.gws.resources[5].signal_count, 16)

    def test_bulk_release_partial_waiters(self) -> None:
        rid = 8
        for w in [10, 20, 30, 40, 50]:
            self.gws.wait_p(rid, wave_id=w)

        # Bulk release count 3 releases first 3 waiters in FIFO order
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=3)
        self.assertEqual(awakened, [10, 20, 30])
        self.assertEqual(val, 0)
        self.assertEqual(self.gws.resources[rid].wait_queue, [40, 50])

    def test_bulk_release_exact_waiters(self) -> None:
        rid = 12
        for w in [1, 2, 3]:
            self.gws.wait_p(rid, wave_id=w)

        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=3)
        self.assertEqual(awakened, [1, 2, 3])
        self.assertEqual(val, 0)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_bulk_release_exceeding_waiters_accumulates_counter(self) -> None:
        rid = 15
        for w in [101, 102]:
            self.gws.wait_p(rid, wave_id=w)

        # Release count 5 wakes 2 waiters, remaining 3 permits increment counter
        val, awakened = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=5)
        self.assertEqual(awakened, [101, 102])
        self.assertEqual(val, 3)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)

    def test_counter_wrapping_modulo_32bit(self) -> None:
        rid = 20
        self.gws.init_resource(rid, init_val=0xFFFF_FFF0)
        val, _ = ref_ds_gws_sema_br(self.gws, m0_val=rid, offset=0, release_count=20)
        # 0xFFFFFFF0 + 20 wraps to 4 modulo 2^32
        self.assertEqual(val, 4)
        self.assertEqual(self.gws.resources[rid].value, 4)

    def test_resource_id_wrapping(self) -> None:
        self.assertEqual(ref_gws_resource_id(m0_val=62, offset=2), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=8), 4)

    def test_resource_isolation(self) -> None:
        ref_ds_gws_sema_br(self.gws, m0_val=1, offset=0, release_count=5)
        ref_ds_gws_sema_br(self.gws, m0_val=2, offset=0, release_count=10)

        self.assertEqual(self.gws.resources[1].value, 5)
        self.assertEqual(self.gws.resources[2].value, 10)
        self.assertEqual(self.gws.resources[0].value, 0)


class TestGWSSemaBrAST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions for ds_gws_sema_br."""

    def test_register_construction_and_formatting(self) -> None:
        v0 = Register(0, RegKind.VGPR)
        v255 = Register(255, RegKind.VGPR)
        s15 = Register(15, RegKind.SGPR)
        m0 = Register(0, RegKind.M0)

        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        self.assertEqual(str(s15), "s15")
        self.assertEqual(str(m0), "m0")

    def test_register_bounds_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Register(-1, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(256, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(1, RegKind.M0)

    def test_instruction_disassembly_formatting(self) -> None:
        # Standard form without vdata
        i1 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR, offset=0, gds=True)
        self.assertEqual(i1.disasm(), "ds_gws_sema_br offset:0 gds")

        # Form with vdata and offset
        i2 = Instruction(opcode=Opcode.DS_GWS_SEMA_BR, vdata=Register(0, RegKind.VGPR), offset=4, gds=True)
        self.assertEqual(i2.disasm(), "ds_gws_sema_br v0 offset:4 gds")

    def test_instruction_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(offset=-1)
        with self.assertRaises(ValueError):
            Instruction(offset=256)

    def test_gws_expression_symbolic_evaluation(self) -> None:
        expr = GWSExpression(offset=8)
        self.assertEqual(expr.evaluate(m0_val=0), 8)
        self.assertEqual(expr.evaluate(m0_val=60), 4)


class TestGWSSemaBrScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter for ds_gws_sema_br."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_step_default_count(self) -> None:
        self.interp.rf.m0 = 5
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_BR, offset=3)
        val, awakened = self.interp.step(instr)
        self.assertEqual(val, 1)
        self.assertEqual(awakened, [])
        self.assertEqual(self.interp.gws.resources[8].value, 1)

    def test_scalar_step_with_vdata_count(self) -> None:
        self.interp.rf.m0 = 0
        self.interp.rf.set_vgpr(1, 25)
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_BR, vdata=Register(1, RegKind.VGPR), offset=7)
        val, _ = self.interp.step(instr)
        self.assertEqual(val, 25)
        self.assertEqual(self.interp.gws.resources[7].value, 25)

    def test_scalar_step_awakens_waiters(self) -> None:
        rid = 10
        self.interp.gws.wait_p(rid, wave_id=12)
        self.interp.gws.wait_p(rid, wave_id=13)
        self.interp.rf.m0 = 10
        self.interp.rf.set_vgpr(0, 4)

        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_BR, vdata=Register(0, RegKind.VGPR), offset=0)
        val, awakened = self.interp.step(instr)
        self.assertEqual(awakened, [12, 13])
        self.assertEqual(val, 2)  # 4 permits - 2 waiters = 2 left in counter

    def test_scalar_step_trace_determinism(self) -> None:
        self.interp.rf.m0 = 1
        self.interp.rf.set_vgpr(2, 5)
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_BR, vdata=Register(2, RegKind.VGPR), offset=0)
        for i in range(1, 4):
            val, _ = self.interp.step(instr)
            self.assertEqual(val, i * 5)
        self.assertEqual(len(self.interp.trace), 3)


if __name__ == "__main__":
    unittest.main()
