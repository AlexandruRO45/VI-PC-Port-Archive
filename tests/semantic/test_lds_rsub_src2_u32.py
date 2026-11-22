"""Semantic reference oracle and architectural model for ds_rsub_src2_u32.

RDNA2 / GFX10 Local Data Share (LDS) Dual-Source Atomic Reverse Subtraction:
Performs a 32-bit atomic reverse subtraction of secondary operand LDS[B] minus primary operand LDS[A],
storing the result back into LDS[A]:
  tmp_a = LDS[A]
  tmp_b = LDS[B]
  LDS[A] = (tmp_b - tmp_a) mod 2^32

Syntax:
    ds_rsub_src2_u32 vaddr [offset0:offset8] [offset1:offset8]

Semantics:
    Address A = (vaddr + offset0 * 4) & 0xFFFFFFFF
    Address B = (vaddr + offset1 * 4) & 0xFFFFFFFF
    idx_a = (Address A >> 2) & (lds_dwords - 1)
    idx_b = (Address B >> 2) & (lds_dwords - 1)
    val_a = LDS[idx_a]
    val_b = LDS[idx_b]
    LDS[idx_a] = (val_b - val_a) & 0xFFFFFFFF

Non-returning:
    This instruction does not write to vector general-purpose registers (VGPRs).

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import struct
import sys
from typing import Any, Tuple
import unittest

# Architectural Constants for RDNA2 Local Data Share (LDS)
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK32: int = 0xFFFFFFFF
OFFSET8_MASK: int = 0xFF
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def calc_lds_src2_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    capacity_dwords: int = LDS_SIZE_DWORDS,
) -> tuple[int, int, int, int]:
    """Calculate effective byte addresses and dword indices for LDS Dual-Source operands A and B."""
    byte_a = (vaddr + ((offset0 & OFFSET8_MASK) * 4)) & MASK32
    idx_a = (byte_a >> 2) & (capacity_dwords - 1)

    byte_b = (vaddr + ((offset1 & OFFSET8_MASK) * 4)) & MASK32
    idx_b = (byte_b >> 2) & (capacity_dwords - 1)

    return byte_a, idx_a, byte_b, idx_b


def oracle_ds_rsub_src2_u32(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_rsub_src2_u32.
    
    Reverse subtraction: res = (val_b - val_a) mod 2^32.
    """
    a = val_a & MASK32
    b = val_b & MASK32
    return (b - a) & MASK32


class LDSModel:
    """Architectural memory model representing the RDNA2 Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords: int = capacity_dwords
        self.memory: list[int] = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        self.memory[idx] = val & MASK32

    def atomic_rsub_src2_u32(self, vaddr: int, offset0: int = 0, offset1: int = 0) -> int:
        """Execute atomic dual-source reverse subtraction on LDS."""
        byte_a, idx_a, byte_b, idx_b = calc_lds_src2_addresses(
            vaddr, offset0, offset1, self.capacity_dwords
        )
        val_a = self.memory[idx_a]
        val_b = self.memory[idx_b]
        res = oracle_ds_rsub_src2_u32(val_a, val_b)
        self.memory[idx_a] = res
        return res


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand representation."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2, 4):
            raise ValueError(f"Unsupported register count: {self.count}")
        if self.index + self.count > 256:
            raise ValueError(f"Register range exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count > 1:
            return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class DSSrc2Offset:
    """Immediate byte offset modifiers for DS dual-source instructions."""
    offset0: int = 0
    offset1: int = 0

    def __post_init__(self) -> None:
        if self.offset0 < 0 or self.offset0 > 255:
            raise ValueError(f"offset0 must be in unsigned 8-bit range [0, 255]: {self.offset0}")
        if self.offset1 < 0 or self.offset1 > 255:
            raise ValueError(f"offset1 must be in unsigned 8-bit range [0, 255]: {self.offset1}")


class Opcode(enum.Enum):
    """LDS dual-source instruction opcodes."""
    DS_RSUB_SRC2_U32 = "ds_rsub_src2_u32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a ds_rsub_src2_u32 instruction."""
    opcode: Opcode
    vaddr: Register
    offset: DSSrc2Offset = dataclasses.field(default_factory=DSSrc2Offset)

    def __post_init__(self) -> None:
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR (count=1): {self.vaddr}")

    def to_assembly(self) -> str:
        va_str = self.vaddr.format_asm()
        off_parts: list[str] = []
        if self.offset.offset0 != 0:
            off_parts.append(f"offset0:{self.offset.offset0}")
        if self.offset.offset1 != 0:
            off_parts.append(f"offset1:{self.offset.offset1}")
        off_str = (" " + " ".join(off_parts)) if off_parts else ""
        return f"{self.opcode.value} {va_str}{off_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of LDS dual-source instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LdsRsubSrc2ConditionCodes:
    """Condition codes resulting from reverse subtraction."""
    zero: bool
    negative: bool
    positive: bool
    borrow_wrapped: bool


@dataclasses.dataclass(frozen=True)
class LdsRsubSrc2StatusFlags:
    """Execution status and address resolution metadata."""
    byte_address_a: int
    idx_a: int
    byte_address_b: int
    idx_b: int
    old_val_a: int
    val_b: int
    new_val_a: int

class ScalarLDSInterpreter:
    """Deterministic scalar interpreter for ds_rsub_src2_u32."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.lds: LDSModel = LDSModel(capacity_dwords)
        self.vcc: int = 0
        self.last_condition_codes: Any | None = None
        self.last_status_flags: Any | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= len(self.vgprs)) or (kind == RegKind.SGPR and idx >= len(self.sgprs)):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= len(self.vgprs)) or (kind == RegKind.SGPR and idx >= len(self.sgprs)):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def step(self, inst: Instruction) -> int:
        """Executes a single ds_rsub_src2_u32 instruction."""
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        byte_a, idx_a, byte_b, idx_b = calc_lds_src2_addresses(
            vaddr, inst.offset.offset0, inst.offset.offset1, self.lds.capacity_dwords
        )
        old_val_a = self.lds.memory[idx_a]
        val_b = self.lds.memory[idx_b]
        res = self.lds.atomic_rsub_src2_u32(
            vaddr=vaddr,
            offset0=inst.offset.offset0,
            offset1=inst.offset.offset1,
        )
        borrow = (val_b < old_val_a)
        self.last_condition_codes = LdsRsubSrc2ConditionCodes(
            zero=(res == 0),
            negative=bool(res & SIGN_BIT32),
            positive=(res > 0 and not bool(res & SIGN_BIT32)),
            borrow_wrapped=borrow,
        )
        self.last_status_flags = LdsRsubSrc2StatusFlags(
            byte_address_a=byte_a,
            idx_a=idx_a,
            byte_address_b=byte_b,
            idx_b=idx_b,
            old_val_a=old_val_a,
            val_b=val_b,
            new_val_a=res,
        )
        self.trace.append(f"{inst.to_assembly()} => res={res:#010x}")
        return res

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDLDSWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for LDS dual-source atomics."""

    def __init__(self, wave_size: int = 64, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.lds: LDSModel = LDSModel(capacity_dwords)

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][idx] = val32
        else:
            self.sgprs[idx] = val32

    def step(self, inst: Instruction) -> list[int | None]:
        """Executes a ds_rsub_src2_u32 instruction across all lanes in the wavefront."""
        # 1. Latch inputs for each lane
        lane_inputs: list[int] = []
        for lane in range(self.wave_size):
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append(va)

        # 2. Serialize atomic operations across active lanes
        lane_results: list[int | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            vaddr = lane_inputs[lane]
            res = self.lds.atomic_rsub_src2_u32(
                vaddr=vaddr,
                offset0=inst.offset.offset0,
                offset1=inst.offset.offset1,
            )
            lane_results.append(res)

        return lane_results


class TestLdsRsubSrc2SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, capacity_dwords=1024)
        for lane in range(32):
            # slot A at lane * 8, slot B at lane * 8 + 4
            simd.lds.write_dword(lane * 8, 10)
            simd.lds.write_dword(lane * 8 + 4, 100 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 8)

        # offset0 = 0 (byte 0), offset1 = 1 (byte 4)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=0, offset1=1),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], 90 + lane)
            self.assertEqual(simd.lds.read_dword(lane * 8), 90 + lane)
            self.assertEqual(simd.lds.read_dword(lane * 8 + 4), 100 + lane)

    def test_wave64_convergent_shared_address(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=64, capacity_dwords=1024)
        # Shared slot A at address 0
        simd.lds.write_dword(0, 0)
        # Each lane i uses slot B at address (i + 1) * 4
        for lane in range(64):
            simd.lds.write_dword((lane + 1) * 4, (lane + 1) * 10)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        curr_a = 0
        for lane in range(64):
            inst = Instruction(
                opcode=Opcode.DS_RSUB_SRC2_U32,
                vaddr=Register(RegKind.VGPR, 1),
                offset=DSSrc2Offset(offset0=0, offset1=lane + 1),
            )
            # Run one lane active test
            simd.set_exec_mask(1 << lane)
            results = simd.step(inst)
            b_val = (lane + 1) * 10
            expected_a = (b_val - curr_a) & MASK32
            self.assertEqual(results[lane], expected_a)
            curr_a = expected_a

        self.assertEqual(simd.lds.read_dword(0), curr_a)

    def test_wave32_exec_masking_partial_lanes(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, capacity_dwords=1024)
        simd.set_exec_mask(0x000000FF)  # Only lanes 0..7 active

        for lane in range(32):
            simd.lds.write_dword(lane * 8, 20)
            simd.lds.write_dword(lane * 8 + 4, 100)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 8)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=0, offset1=1),
        )
        results = simd.step(inst)

        for lane in range(32):
            if lane < 8:
                self.assertEqual(results[lane], 80)
                self.assertEqual(simd.lds.read_dword(lane * 8), 80)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(simd.lds.read_dword(lane * 8), 20)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=64, capacity_dwords=1024)
        simd.set_exec_mask(0)  # All inactive
        simd.lds.write_dword(0, 555)
        simd.lds.write_dword(4, 999)

        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=0, offset1=1),
        )
        results = simd.step(inst)
        self.assertEqual(results, [None] * 64)
        self.assertEqual(simd.lds.read_dword(0), 555)
        self.assertEqual(simd.lds.read_dword(4), 999)


class TestLdsRsubSrc2ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_scalar_execution_basic(self) -> None:
        # Seed slot A (0x10) with 30, slot B (0x20) with 100
        self.interp.lds.write_dword(0x10, 30)
        self.interp.lds.write_dword(0x20, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x00)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=4, offset1=8),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 70)
        self.assertEqual(self.interp.lds.read_dword(0x10), 70)
        self.assertEqual(self.interp.lds.read_dword(0x20), 100)
        # VGPR v1 is preserved (non-returning instruction)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 0x00)

    def test_scalar_chained_execution(self) -> None:
        # Step 1: 0x10 has 10, 0x20 has 50 -> 0x10 becomes 50 - 10 = 40
        # Step 2: 0x10 has 40, 0x30 has 100 -> 0x10 becomes 100 - 40 = 60
        self.interp.lds.write_dword(0x10, 10)
        self.interp.lds.write_dword(0x20, 50)
        self.interp.lds.write_dword(0x30, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x00)

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=4, offset1=8),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=4, offset1=12),
        ))

        results = self.interp.run_program(prog)
        self.assertEqual(results, [40, 60])
        self.assertEqual(self.interp.lds.read_dword(0x10), 60)

    def test_scalar_self_addressing(self) -> None:
        self.interp.lds.write_dword(0x40, 0x12345678)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x40)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=0, offset1=0),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 0)
        self.assertEqual(self.interp.lds.read_dword(0x40), 0)


class TestLdsRsubSrc2AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v1 = Register(RegKind.VGPR, 1)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v1.format_asm(), "v1")

    def test_instruction_formatting(self) -> None:
        inst_no_off = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
        )
        self.assertEqual(inst_no_off.to_assembly(), "ds_rsub_src2_u32 v1")

        inst_with_off = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 2),
            offset=DSSrc2Offset(offset0=4, offset1=8),
        )
        self.assertEqual(inst_with_off.to_assembly(), "ds_rsub_src2_u32 v2 offset0:4 offset1:8")

    def test_instruction_validation(self) -> None:
        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_SRC2_U32,
                vaddr=Register(RegKind.SGPR, 1),
            )

        # vaddr cannot be multi-register
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_SRC2_U32,
                vaddr=Register(RegKind.VGPR, 1, count=2),
            )

    def test_offset_validation(self) -> None:
        with self.assertRaises(ValueError):
            DSSrc2Offset(offset0=-1, offset1=0)
        with self.assertRaises(ValueError):
            DSSrc2Offset(offset0=0, offset1=256)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=1, offset1=2),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 3),
        ))
        expected = "ds_rsub_src2_u32 v1 offset0:1 offset1:2\nds_rsub_src2_u32 v3"
        self.assertEqual(prog.to_assembly(), expected)


class TestLdsRsubSrc2ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model of ds_rsub_src2_u32."""

    def setUp(self) -> None:
        self.lds = LDSModel(LDS_SIZE_DWORDS)

    def test_basic_reverse_subtraction(self) -> None:
        # LDS[A] = 30, LDS[B] = 100 -> new LDS[A] = 100 - 30 = 70
        res = oracle_ds_rsub_src2_u32(val_a=30, val_b=100)
        self.assertEqual(res, 70)

    def test_reverse_subtraction_underflow_wrapping(self) -> None:
        # LDS[A] = 100, LDS[B] = 30 -> new LDS[A] = (30 - 100) mod 2^32 = -70 mod 2^32 = 0xFFFFFFBA
        res = oracle_ds_rsub_src2_u32(val_a=100, val_b=30)
        self.assertEqual(res, (30 - 100) & MASK32)
        self.assertEqual(res, 0xFFFFFFBA)

    def test_self_subtraction_zeroes(self) -> None:
        # When A == B (same index), val_a == val_b -> (val_b - val_a) = 0
        for val in (0, 1, 42, 0x12345678, 0xFFFFFFFF):
            self.assertEqual(oracle_ds_rsub_src2_u32(val, val), 0)

    def test_lds_model_dual_source_execution(self) -> None:
        # Seed slot A (byte 0x10) with 25 and slot B (byte 0x20) with 100
        self.lds.write_dword(0x10, 25)
        self.lds.write_dword(0x20, 100)

        # vaddr = 0x00, offset0 = 4 (byte 16 = 0x10), offset1 = 8 (byte 32 = 0x20)
        res = self.lds.atomic_rsub_src2_u32(vaddr=0x00, offset0=4, offset1=8)
        self.assertEqual(res, 75)
        # LDS[A] updated to 75
        self.assertEqual(self.lds.read_dword(0x10), 75)
        # LDS[B] untouched (remains 100)
        self.assertEqual(self.lds.read_dword(0x20), 100)

    def test_self_addressing_in_lds_model(self) -> None:
        # Both offset0 and offset1 point to byte 0x40
        self.lds.write_dword(0x40, 0xABCDEF01)
        res = self.lds.atomic_rsub_src2_u32(vaddr=0x40, offset0=0, offset1=0)
        self.assertEqual(res, 0)
        self.assertEqual(self.lds.read_dword(0x40), 0)

    def test_dword_alignment_drops_low_bits(self) -> None:
        # vaddr = 0x13 -> drops low 2 bits to 0x10
        self.lds.write_dword(0x10, 40)
        self.lds.write_dword(0x18, 90)
        # offset0 = 0 -> address 0x13 -> dword 4 (byte 0x10)
        # offset1 = 2 -> address 0x1B -> dword 6 (byte 0x18)
        res = self.lds.atomic_rsub_src2_u32(vaddr=0x13, offset0=0, offset1=2)
        self.assertEqual(res, 50)
        self.assertEqual(self.lds.read_dword(0x10), 50)
        self.assertEqual(self.lds.read_dword(0x18), 90)

    def test_address_wrapping_modulo_lds_size(self) -> None:
        mem = LDSModel(capacity_dwords=256)  # 256 dwords (1024 bytes)
        mem.write_dword(0x00, 15)
        mem.write_dword(0x20, 50)
        # Byte 1024 wraps to 0x00 (dword 0)
        res = mem.atomic_rsub_src2_u32(vaddr=1024, offset0=0, offset1=8)  # offset1=8 -> byte 32 (0x20 = dword 8)
        self.assertEqual(res, 35)
        self.assertEqual(mem.read_dword(0x00), 35)




class TestLdsRsubSrc2RegisterAliasing(unittest.TestCase):
    """Unit tests for register preservation and non-returning contract."""

    def test_unrelated_vgprs_preserved(self) -> None:
        interp = ScalarLDSInterpreter(capacity_dwords=1024)
        for r in range(256):
            interp.write_reg_u32(RegKind.VGPR, r, 0xCAFE0000 | r)
        interp.write_reg_u32(RegKind.VGPR, 5, 0x100)
        interp.lds.write_dword(0x100, 20)
        interp.lds.write_dword(0x104, 50)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 5),
            offset=DSSrc2Offset(offset0=0, offset1=1),
        )
        res = interp.step(inst)
        self.assertEqual(res, 30)
        for r in range(256):
            expected = 0x100 if r == 5 else (0xCAFE0000 | r)
            self.assertEqual(interp.read_reg_u32(RegKind.VGPR, r), expected)

    def test_vaddr_preservation(self) -> None:
        interp = ScalarLDSInterpreter(capacity_dwords=1024)
        interp.write_reg_u32(RegKind.VGPR, 10, 0x1234)
        interp.lds.write_dword(0x1234, 10)
        interp.lds.write_dword(0x1238, 90)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 10),
            offset=DSSrc2Offset(offset0=0, offset1=1),
        )
        interp.step(inst)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 10), 0x1234)

    def test_simd_shared_vaddr_across_divergent_lanes(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, capacity_dwords=1024)
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 3, 0x200)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 4, 0xDEAD0000 | lane)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 3),
            offset=DSSrc2Offset(offset0=0, offset1=1),
        )
        simd.lds.write_dword(0x200, 10)
        simd.lds.write_dword(0x204, 100)
        simd.step(inst)
        for lane in range(32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 3), 0x200)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 4), 0xDEAD0000 | lane)


class TestLdsRsubSrc2ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_zero_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 50)
        self.interp.lds.write_dword(0x20, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x00)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=4, offset1=8),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 0)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)
        self.assertFalse(cc.borrow_wrapped)

    def test_positive_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 40)
        self.interp.lds.write_dword(0x20, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x00)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=4, offset1=8),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 60)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        self.assertFalse(cc.zero)
        self.assertFalse(cc.negative)
        self.assertTrue(cc.positive)
        self.assertFalse(cc.borrow_wrapped)

    def test_negative_and_borrow_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 50)
        self.interp.lds.write_dword(0x20, 40)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x00)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=4, offset1=8),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 0xFFFFFFF6)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.negative)
        self.assertFalse(cc.positive)
        self.assertTrue(cc.borrow_wrapped)

    def test_vcc_isolation(self) -> None:
        self.interp.vcc = 0xDEADBEEF
        self.interp.lds.write_dword(0x10, 10)
        self.interp.lds.write_dword(0x20, 20)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSSrc2Offset(offset0=4, offset1=8),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, 0xDEADBEEF)

    def test_status_flags_metadata(self) -> None:
        self.interp.lds.write_dword(0x10, 15)
        self.interp.lds.write_dword(0x24, 75)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0x10)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_SRC2_U32,
            vaddr=Register(RegKind.VGPR, 2),
            offset=DSSrc2Offset(offset0=0, offset1=5),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 60)
        sf = self.interp.last_status_flags
        self.assertIsNotNone(sf)
        self.assertEqual(sf.byte_address_a, 0x10)
        self.assertEqual(sf.idx_a, 4)
        self.assertEqual(sf.byte_address_b, 0x24)
        self.assertEqual(sf.idx_b, 9)
        self.assertEqual(sf.old_val_a, 15)
        self.assertEqual(sf.val_b, 75)
        self.assertEqual(sf.new_val_a, 60)

if __name__ == "__main__":
    unittest.main()
