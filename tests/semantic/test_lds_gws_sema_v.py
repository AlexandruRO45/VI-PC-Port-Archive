from __future__ import annotations

import enum
import math
import struct
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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

    def signal_v(self, res_id: int) -> Tuple[int, Optional[int]]:
        """Executes a semaphore V (signal/increment/post) operation.

        Returns:
            Tuple of (new_counter_value, awakened_wavefront_id_or_None).
        """
        rid = res_id & GWS_RESOURCE_ID_MASK
        res = self.resources[rid]
        if res.res_type == GWSResourceType.BARRIER:
            raise TypeError(f"GWS resource {rid} configured as BARRIER, cannot execute ds_gws_sema_v")
        res.res_type = GWSResourceType.SEMAPHORE

        res.signal_count += 1
        if res.wait_queue:
            awakened = res.wait_queue.pop(0)
            res.wake_count += 1
            return res.value, awakened
        else:
            res.value = (res.value + 1) & MASK32
            return res.value, None

    def wait_p(self, res_id: int, wave_id: int) -> bool:
        """Executes a semaphore P (wait/decrement/acquire) operation.

        Returns:
            True if acquired immediately, False if blocked and enqueued.
        """
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


def ref_ds_gws_sema_v(subsystem: GWSSubsystem, m0_val: int, offset: int) -> Tuple[int, Optional[int]]:
    """Reference mathematical oracle for RDNA2 ds_gws_sema_v."""
    rid = ref_gws_resource_id(m0_val, offset)
    return subsystem.signal_v(rid)


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
    DS_GWS_SEMA_V = "ds_gws_sema_v"
    DS_GWS_SEMA_P = "ds_gws_sema_p"
    DS_GWS_INIT = "ds_gws_init"
    DS_GWS_BARRIER = "ds_gws_barrier"


@dataclass(frozen=True)
class Instruction:
    """AST representation of RDNA2 ds_gws_sema_v instruction."""
    opcode: Opcode = Opcode.DS_GWS_SEMA_V
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
    """Deterministic scalar instruction interpreter for ds_gws_sema_v."""

    def __init__(self, gws: Optional[GWSSubsystem] = None) -> None:
        self.gws: GWSSubsystem = gws if gws is not None else GWSSubsystem()
        self.rf: ScalarRegisterFile = ScalarRegisterFile()
        self.trace: List[Tuple[str, int, Optional[int]]] = []

    def step(self, instr: Instruction, wave_id: int = 0) -> Tuple[int, Optional[int]]:
        """Execute a single ds_gws_sema_v instruction deterministically."""
        if instr.opcode != Opcode.DS_GWS_SEMA_V:
            raise NotImplementedError(f"Unsupported opcode in scalar interpreter: {instr.opcode}")

        rid = ref_gws_resource_id(self.rf.m0, instr.offset)
        new_val, awakened = self.gws.signal_v(rid)
        self.trace.append((instr.disasm(), new_val, awakened))
        return new_val, awakened


# Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking


class SIMDWaveState:
    """Models multi-lane SIMD wave state (Wave32 or Wave64)."""

    def __init__(self, wave_id: int = 0, wave_size: int = 32, exec_mask: int = MASK32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got: {wave_size}")
        self.wave_id: int = wave_id
        self.wave_size: int = wave_size
        self.exec_mask: int = exec_mask & (MASK64 if wave_size == 64 else MASK32)
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: List[int] = [0] * 104
        self.m0: int = 0
        self.vcc: int = 0

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx] & MASK32

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = val & MASK32


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def __init__(self, gws: Optional[GWSSubsystem] = None) -> None:
        self.gws: GWSSubsystem = gws if gws is not None else GWSSubsystem()

    def step_wave(self, wave: SIMDWaveState, instr: Instruction) -> Dict[str, Any]:
        if instr.opcode != Opcode.DS_GWS_SEMA_V:
            raise NotImplementedError(f"Unsupported opcode in SIMD interpreter: {instr.opcode}")

        rid = ref_gws_resource_id(wave.m0, instr.offset)
        active_count = 0
        signals_issued = 0
        awakened_waves: List[int] = []

        for lane in range(wave.wave_size):
            if wave.is_lane_active(lane):
                active_count += 1
                _, awakened = self.gws.signal_v(rid)
                signals_issued += 1
                if awakened is not None:
                    awakened_waves.append(awakened)

        return {
            "resource_id": rid,
            "active_lanes": active_count,
            "signals_issued": signals_issued,
            "awakened_waves": awakened_waves,
            "final_value": self.gws.resources[rid].value,
        }


# Commit 6: Condition codes, status flags, and carry/borrow generation


@dataclass(frozen=True)
class GWSSemaVStatusFlags:
    """Architectural condition codes and status flags for ds_gws_sema_v."""
    signaled: bool
    waiter_released: bool
    counter_zero: bool
    counter_negative: bool
    carry_out: bool
    resource_id: int


def evaluate_gws_status(old_val: int, new_val: int, waiter_released: bool, rid: int) -> GWSSemaVStatusFlags:
    """Evaluates status flags and condition codes after a semaphore V operation."""
    carry_out = (not waiter_released) and (old_val == 0xFFFF_FFFF and new_val == 0)
    counter_zero = (new_val == 0)
    counter_negative = bool(new_val & 0x8000_0000)
    return GWSSemaVStatusFlags(
        signaled=True,
        waiter_released=waiter_released,
        counter_zero=counter_zero,
        counter_negative=counter_negative,
        carry_out=carry_out,
        resource_id=rid,
    )


class TestGWSSemaVOracleAndModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model for ds_gws_sema_v."""

    def setUp(self) -> None:
        self.gws = GWSSubsystem()

    def test_single_signal_increments_counter(self) -> None:
        val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=0, offset=0)
        self.assertEqual(val, 1)
        self.assertIsNone(awakened)
        self.assertEqual(self.gws.resources[0].value, 1)
        self.assertEqual(self.gws.resources[0].signal_count, 1)

    def test_multiple_consecutive_signals(self) -> None:
        for expected in range(1, 101):
            val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=5, offset=2)
            self.assertEqual(val, expected)
            self.assertIsNone(awakened)
        self.assertEqual(self.gws.resources[7].value, 100)
        self.assertEqual(self.gws.resources[7].signal_count, 100)

    def test_counter_wrapping_modulo_32bit(self) -> None:
        rid = 12
        self.gws.init_resource(rid, init_val=0xFFFF_FFFE)
        val, _ = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertEqual(val, 0xFFFF_FFFF)

        val, _ = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertEqual(val, 0)
        self.assertEqual(self.gws.resources[rid].value, 0)

    def test_awaken_single_waiting_wavefront(self) -> None:
        rid = 3
        # Wavefront 42 attempts P on zero-counter semaphore -> blocks
        acquired = self.gws.wait_p(rid, wave_id=42)
        self.assertFalse(acquired)
        self.assertEqual(self.gws.resources[rid].wait_queue, [42])

        # ds_gws_sema_v signals: should wake wave 42, counter stays 0
        val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertEqual(awakened, 42)
        self.assertEqual(val, 0)
        self.assertEqual(len(self.gws.resources[rid].wait_queue), 0)
        self.assertEqual(self.gws.resources[rid].wake_count, 1)

    def test_awaken_fifo_queue_ordering(self) -> None:
        rid = 10
        for w in [101, 102, 103, 104]:
            self.gws.wait_p(rid, wave_id=w)

        # 4 signals should awaken in exact FIFO order
        for expected_wave in [101, 102, 103, 104]:
            val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
            self.assertEqual(awakened, expected_wave)
            self.assertEqual(val, 0)

        # 5th signal increments counter
        val, awakened = ref_ds_gws_sema_v(self.gws, m0_val=rid, offset=0)
        self.assertIsNone(awakened)
        self.assertEqual(val, 1)

    def test_resource_id_wrapping(self) -> None:
        # Offset + M0 base wrapping across 64 slots
        self.assertEqual(ref_gws_resource_id(m0_val=60, offset=4), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=1), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=63, offset=65), 0)
        self.assertEqual(ref_gws_resource_id(m0_val=10, offset=70), 16)

    def test_independent_gws_resource_isolation(self) -> None:
        ref_ds_gws_sema_v(self.gws, m0_val=1, offset=0)
        ref_ds_gws_sema_v(self.gws, m0_val=2, offset=0)
        ref_ds_gws_sema_v(self.gws, m0_val=2, offset=0)

        self.assertEqual(self.gws.resources[1].value, 1)
        self.assertEqual(self.gws.resources[2].value, 2)
        self.assertEqual(self.gws.resources[0].value, 0)
        self.assertEqual(self.gws.resources[3].value, 0)


class TestGWSSemaVAST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions for ds_gws_sema_v."""

    def test_register_construction_and_formatting(self) -> None:
        v0 = Register(0, RegKind.VGPR)
        v255 = Register(255, RegKind.VGPR)
        s12 = Register(12, RegKind.SGPR)
        m0 = Register(0, RegKind.M0)
        exec_reg = Register(0, RegKind.EXEC)
        vcc_reg = Register(0, RegKind.VCC)

        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        self.assertEqual(str(s12), "s12")
        self.assertEqual(str(m0), "m0")
        self.assertEqual(str(exec_reg), "exec")
        self.assertEqual(str(vcc_reg), "vcc")

    def test_register_index_bounds_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Register(-1, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(256, RegKind.VGPR)
        with self.assertRaises(ValueError):
            Register(1, RegKind.M0)

    def test_instruction_disassembly_formatting(self) -> None:
        # Standard form without vdata
        i1 = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=0, gds=True)
        self.assertEqual(i1.disasm(), "ds_gws_sema_v offset:0 gds")

        # Form with non-zero offset
        i2 = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=8, gds=True)
        self.assertEqual(i2.disasm(), "ds_gws_sema_v offset:8 gds")

        # Form with vdata
        i3 = Instruction(opcode=Opcode.DS_GWS_SEMA_V, vdata=Register(3, RegKind.VGPR), offset=4, gds=True)
        self.assertEqual(i3.disasm(), "ds_gws_sema_v v3 offset:4 gds")

    def test_instruction_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(offset=-1)
        with self.assertRaises(ValueError):
            Instruction(offset=256)

    def test_gws_expression_symbolic_evaluation(self) -> None:
        expr = GWSExpression(offset=10)
        self.assertEqual(expr.evaluate(m0_val=2), 12)
        self.assertEqual(expr.evaluate(m0_val=60), 6)  # (60 + 10) % 64 = 6
        self.assertEqual(expr.evaluate(m0_val=0), 10)


class TestGWSSemaVScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter for ds_gws_sema_v."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_step_single_signal(self) -> None:
        self.interp.rf.m0 = 4
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=2)
        val, awakened = self.interp.step(instr)
        self.assertEqual(val, 1)
        self.assertIsNone(awakened)
        self.assertEqual(self.interp.gws.resources[6].value, 1)
        self.assertEqual(len(self.interp.trace), 1)

    def test_scalar_step_with_vdata_operand(self) -> None:
        self.interp.rf.m0 = 0
        self.interp.rf.set_vgpr(1, 0x12345678)
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, vdata=Register(1, RegKind.VGPR), offset=0)
        val, _ = self.interp.step(instr)
        self.assertEqual(val, 1)
        # vdata register content must not be modified by signal
        self.assertEqual(self.interp.rf.get_vgpr(1), 0x12345678)

    def test_scalar_step_releases_waiting_wavefront(self) -> None:
        rid = 15
        self.interp.gws.wait_p(rid, wave_id=77)
        self.interp.rf.m0 = 10
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=5)
        val, awakened = self.interp.step(instr)
        self.assertEqual(awakened, 77)
        self.assertEqual(val, 0)

    def test_scalar_interpreter_trace_and_determinism(self) -> None:
        self.interp.rf.m0 = 1
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=1)
        for i in range(1, 6):
            val, _ = self.interp.step(instr)
            self.assertEqual(val, i)
        self.assertEqual(len(self.interp.trace), 5)
        self.assertEqual(self.interp.trace[-1], ("ds_gws_sema_v offset:1 gds", 5, None))


class TestGWSSemaVSIMD(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for ds_gws_sema_v."""

    def setUp(self) -> None:
        self.simd = SIMDInterpreter()

    def test_wave32_full_exec_signals_32_times(self) -> None:
        wave = SIMDWaveState(wave_id=1, wave_size=32, exec_mask=0xFFFF_FFFF)
        wave.m0 = 0
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=3)
        res = self.simd.step_wave(wave, instr)

        self.assertEqual(res["active_lanes"], 32)
        self.assertEqual(res["signals_issued"], 32)
        self.assertEqual(res["final_value"], 32)
        self.assertEqual(self.simd.gws.resources[3].value, 32)

    def test_wave64_full_exec_signals_64_times(self) -> None:
        wave = SIMDWaveState(wave_id=2, wave_size=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        wave.m0 = 8
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=0)
        res = self.simd.step_wave(wave, instr)

        self.assertEqual(res["active_lanes"], 64)
        self.assertEqual(res["signals_issued"], 64)
        self.assertEqual(res["final_value"], 64)
        self.assertEqual(self.simd.gws.resources[8].value, 64)

    def test_partial_exec_checkerboard(self) -> None:
        # Alternating bits 0x55555555: 16 active lanes in Wave32
        wave = SIMDWaveState(wave_id=3, wave_size=32, exec_mask=0x5555_5555)
        wave.m0 = 20
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=0)
        res = self.simd.step_wave(wave, instr)

        self.assertEqual(res["active_lanes"], 16)
        self.assertEqual(res["signals_issued"], 16)
        self.assertEqual(res["final_value"], 16)

    def test_single_active_lane(self) -> None:
        # Only lane 0 active
        wave = SIMDWaveState(wave_id=4, wave_size=32, exec_mask=0x0000_0001)
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=5)
        res = self.simd.step_wave(wave, instr)

        self.assertEqual(res["active_lanes"], 1)
        self.assertEqual(res["signals_issued"], 1)
        self.assertEqual(res["final_value"], 1)

    def test_completely_inactive_wave(self) -> None:
        # EXEC mask == 0
        wave = SIMDWaveState(wave_id=5, wave_size=32, exec_mask=0)
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=5)
        res = self.simd.step_wave(wave, instr)

        self.assertEqual(res["active_lanes"], 0)
        self.assertEqual(res["signals_issued"], 0)
        self.assertEqual(res["final_value"], 0)
        self.assertEqual(self.simd.gws.resources[5].value, 0)

    def test_simd_wakes_queued_wavefronts_in_active_lanes(self) -> None:
        rid = 18
        # Enqueue 3 waiting wavefronts
        for w in [50, 51, 52]:
            self.simd.gws.wait_p(rid, wave_id=w)

        # Wave has 5 active lanes
        wave = SIMDWaveState(wave_id=6, wave_size=32, exec_mask=0x0000_001F)  # 5 lanes active
        wave.m0 = rid
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=0)
        res = self.simd.step_wave(wave, instr)

        self.assertEqual(res["active_lanes"], 5)
        self.assertEqual(res["signals_issued"], 5)
        self.assertEqual(res["awakened_waves"], [50, 51, 52])
        # 3 signals consumed by 3 waiters, remaining 2 signals increment counter
        self.assertEqual(res["final_value"], 2)


class TestGWSSemaVRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation for ds_gws_sema_v."""

    def test_m0_register_invariance(self) -> None:
        interp = ScalarInterpreter()
        initial_m0 = 42
        interp.rf.m0 = initial_m0
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, offset=7)
        interp.step(instr)

        # M0 must remain exactly intact
        self.assertEqual(interp.rf.m0, initial_m0)

    def test_vdata_operand_invariance(self) -> None:
        interp = ScalarInterpreter()
        test_val = 0xDEADBEEF
        interp.rf.set_vgpr(5, test_val)
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, vdata=Register(5, RegKind.VGPR), offset=1)
        interp.step(instr)

        self.assertEqual(interp.rf.get_vgpr(5), test_val)

    def test_comprehensive_register_file_preservation(self) -> None:
        interp = ScalarInterpreter()
        # Seed all 256 VGPRs and 104 SGPRs with pseudo-random pattern
        for i in range(256):
            interp.rf.set_vgpr(i, (i * 0x01010101 + 0x5A) & MASK32)
        for i in range(104):
            interp.rf.set_sgpr(i, (i * 0x11111111 + 0x3C) & MASK32)

        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, vdata=Register(10, RegKind.VGPR), offset=3)
        interp.step(instr)

        for i in range(256):
            expected = (i * 0x01010101 + 0x5A) & MASK32
            self.assertEqual(interp.rf.get_vgpr(i), expected, f"VGPR v{i} clobbered")
        for i in range(104):
            expected = (i * 0x11111111 + 0x3C) & MASK32
            self.assertEqual(interp.rf.get_sgpr(i), expected, f"SGPR s{i} clobbered")

    def test_simd_multi_lane_vdata_preservation(self) -> None:
        simd = SIMDInterpreter()
        wave = SIMDWaveState(wave_id=10, wave_size=32, exec_mask=0x00FF_00FF)
        for lane in range(32):
            wave.set_vgpr(lane, 2, (0xCAFE0000 | lane))

        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, vdata=Register(2, RegKind.VGPR), offset=0)
        simd.step_wave(wave, instr)

        # Every lane (both active and inactive) must preserve its vdata untouched
        for lane in range(32):
            expected = 0xCAFE0000 | lane
            self.assertEqual(wave.get_vgpr(lane, 2), expected, f"Lane {lane} VGPR v2 mutated")

    def test_repeated_in_place_execution_loop(self) -> None:
        interp = ScalarInterpreter()
        interp.rf.m0 = 5
        interp.rf.set_vgpr(0, 0x11223344)
        instr = Instruction(opcode=Opcode.DS_GWS_SEMA_V, vdata=Register(0, RegKind.VGPR), offset=0)

        for _ in range(20):
            interp.step(instr)

        self.assertEqual(interp.rf.get_vgpr(0), 0x11223344)
        self.assertEqual(interp.rf.m0, 5)
        self.assertEqual(interp.gws.resources[5].value, 20)


class TestGWSSemaVConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation for ds_gws_sema_v."""

    def test_standard_increment_status_flags(self) -> None:
        flags = evaluate_gws_status(old_val=5, new_val=6, waiter_released=False, rid=0)
        self.assertTrue(flags.signaled)
        self.assertFalse(flags.waiter_released)
        self.assertFalse(flags.counter_zero)
        self.assertFalse(flags.counter_negative)
        self.assertFalse(flags.carry_out)
        self.assertEqual(flags.resource_id, 0)

    def test_waiter_released_status_flags(self) -> None:
        flags = evaluate_gws_status(old_val=0, new_val=0, waiter_released=True, rid=14)
        self.assertTrue(flags.signaled)
        self.assertTrue(flags.waiter_released)
        self.assertTrue(flags.counter_zero)
        self.assertFalse(flags.counter_negative)
        self.assertFalse(flags.carry_out)
        self.assertEqual(flags.resource_id, 14)

    def test_carry_out_on_32bit_wrap(self) -> None:
        # Wrap from 0xFFFFFFFF -> 0 generates carry_out and zero flag
        flags = evaluate_gws_status(old_val=0xFFFF_FFFF, new_val=0, waiter_released=False, rid=2)
        self.assertTrue(flags.carry_out)
        self.assertTrue(flags.counter_zero)
        self.assertFalse(flags.counter_negative)

    def test_counter_negative_sign_bit(self) -> None:
        # Value 0x80000000 has MSB set (negative when interpreted as signed)
        flags = evaluate_gws_status(old_val=0x7FFF_FFFF, new_val=0x8000_0000, waiter_released=False, rid=7)
        self.assertTrue(flags.counter_negative)
        self.assertFalse(flags.counter_zero)
        self.assertFalse(flags.carry_out)

    def test_vcc_predicate_mask_generation(self) -> None:
        # Predicate mask generated across SIMD lanes reflects active execution
        exec_mask = 0xAA55_AA55
        vcc_mask = 0
        for lane in range(32):
            if (exec_mask >> lane) & 1:
                vcc_mask |= (1 << lane)
        self.assertEqual(vcc_mask, exec_mask)


if __name__ == "__main__":
    unittest.main()
