from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_rsub_rtn_u32.

Models unsigned 32-bit integer reverse-subtraction and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_rsub_rtn_u32: Atomic reverse subtraction of unsigned 32-bit integer in LDS memory,
    returning the previous memory content into destination VGPR.

Syntax:
    ds_rsub_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    new_mem = (vdata0 - old_val) & 0xFFFFFFFF
    LDS[dword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. Reverse Subtraction (RSUB):
     Computes (vdata0 - old_val) rather than (old_val - vdata0).
     The incoming operand vdata0 serves as the minuend, and the current memory content
     serves as the subtrahend.
  2. Return Value Semantics (RTN):
     Atomically fetches the pre-arithmetic value of LDS[dword_index] and returns it
     into the destination VGPR (vdst).
  3. Addressing and Alignment:
     Local Data Share is byte-addressed but dword-aligned for 32-bit operations.
     The bottom 2 bits of the effective address are dropped (eff_addr >> 2).
     Immediate offset is an unsigned 16-bit integer (0 <= offset <= 65535).
  4. LDS Capacity and Banking:
     Standard allocation is 64 KiB (16,384 dwords) organized into 32 memory banks.
     Bank index = dword_index % 32.

Standard library Python only; zero external dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
from dataclasses import dataclass
import enum
from enum import Enum, auto
import re
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import unittest

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536      # 64 KiB default capacity
DEFAULT_LDS_NUM_DWORDS: int = 16384      # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32                  # 32 memory banks
DEFAULT_NUM_VGPRS: int = 256             # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106             # Standard GFX10 scalar register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the lowest 2 bits are dropped by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


def ref_ds_rsub_rtn_u32(current_mem: int, vdata0: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_rsub_rtn_u32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = (vdata0 - old_val) & 0xFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    new_mem = u32(u32(vdata0) - old_val)
    return old_val, new_mem


class LDSMemory:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.dwords = [0] * self.num_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        return self.dwords[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        self.dwords[idx] = u32(val)

    def atomic_rsub_rtn_u32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit integer reverse subtraction with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        curr = self.dwords[idx]
        old_val, new_mem = ref_ds_rsub_rtn_u32(curr, vdata0)
        self.dwords[idx] = new_mem
        return old_val, new_mem


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand."""
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
class DSOffset:
    """Immediate byte offset modifier for DS instructions."""
    offset16: int = 0

    def __post_init__(self) -> None:
        if self.offset16 < 0 or self.offset16 > 65535:
            raise ValueError(f"offset16 must be in unsigned 16-bit range [0, 65535]: {self.offset16}")


class Opcode(enum.Enum):
    """LDS instruction opcodes."""
    DS_RSUB_RTN_U32 = "ds_rsub_rtn_u32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a ds_rsub_rtn_u32 instruction."""
    opcode: Opcode
    vdst: Register
    vaddr: Register
    vdata0: Register
    offset: DSOffset = dataclasses.field(default_factory=DSOffset)

    def __post_init__(self) -> None:
        if not self.vdst.is_vgpr or self.vdst.count != 1:
            raise ValueError(f"vdst must be a single 32-bit VGPR: {self.vdst}")
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR: {self.vaddr}")
        if not self.vdata0.is_vgpr or self.vdata0.count != 1:
            raise ValueError(f"vdata0 must be a single 32-bit VGPR: {self.vdata0}")

    def to_assembly(self) -> str:
        vd_str = self.vdst.format_asm()
        va_str = self.vaddr.format_asm()
        v0_str = self.vdata0.format_asm()
        off_str = f" offset:{self.offset.offset16}" if self.offset.offset16 != 0 else ""
        return f"{self.opcode.value} {vd_str}, {va_str}, {v0_str}{off_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of LDS instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LdsRsubConditionCodes:
    """Architectural integer condition codes evaluating the atomic reverse subtraction result."""
    zero: bool
    negative: bool
    positive: bool
    borrow_wrapped: bool


@dataclasses.dataclass(frozen=True)
class LdsRsubStatusFlags:
    """LDS memory addressing and bank routing status flags."""
    effective_address: int
    dword_index: int
    bank_index: int


def compute_lds_rsub_condition_codes(vdata0: int, old_val: int, new_mem: int) -> LdsRsubConditionCodes:
    """Computes arithmetic condition codes for the reverse subtraction (vdata0 - old_val)."""
    v0 = vdata0 & MASK32
    old = old_val & MASK32
    res = new_mem & MASK32
    signed_res = s32(res)
    borrow = v0 < old
    return LdsRsubConditionCodes(
        zero=(res == 0),
        negative=(signed_res < 0),
        positive=(signed_res > 0),
        borrow_wrapped=borrow,
    )


def compute_lds_rsub_status_flags(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> LdsRsubStatusFlags:
    """Computes LDS address and bank status flags."""
    eff = eff_addr & MASK32
    d_idx = calc_lds_dword_index(eff, capacity_dwords)
    b_idx = calc_lds_bank(eff)
    return LdsRsubStatusFlags(
        effective_address=eff,
        dword_index=d_idx,
        bank_index=b_idx,
    )


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarLDSInterpreter:
    """Deterministic scalar interpreter for ds_rsub_rtn_u32."""

    def __init__(self, lds_size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(lds_size_bytes)
        self.vcc: int = 0
        self.last_condition_codes: LdsRsubConditionCodes | None = None
        self.last_status_flags: LdsRsubStatusFlags | None = None
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

    def step(self, inst: Instruction) -> tuple[int, int]:
        """Executes a single ds_rsub_rtn_u32 instruction."""
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        vdata0 = self.read_reg_u32(RegKind.VGPR, inst.vdata0.index)

        old_val, new_mem = self.lds.atomic_rsub_rtn_u32(
            vaddr=vaddr,
            vdata0=vdata0,
            offset=inst.offset.offset16,
        )

        self.write_reg_u32(RegKind.VGPR, inst.vdst.index, old_val)

        eff_addr = calc_lds_effective_address(vaddr, inst.offset.offset16)
        self.last_condition_codes = compute_lds_rsub_condition_codes(vdata0, old_val, new_mem)
        self.last_status_flags = compute_lds_rsub_status_flags(eff_addr, self.lds.num_dwords)
        # Note: LDS operations never modify scalar condition codes or VCC.

        self.trace.append(f"{inst.to_assembly()} => old={old_val:#010x}, new={new_mem:#010x}")
        return old_val, new_mem

    def run_program(self, prog: Program) -> list[tuple[int, int]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDLDSWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for LDS atomics."""

    def __init__(self, wave_size: int = 64, lds_size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[list[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(lds_size_bytes)

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

    def step(self, inst: Instruction) -> list[tuple[int | None, int | None]]:
        """Executes a ds_rsub_rtn_u32 instruction across all lanes in the wavefront."""
        # 1. Latch inputs for each lane
        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            v0 = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata0.index)
            lane_inputs.append((va, v0))

        # 2. Serialize atomic operations across active lanes
        lane_results: list[tuple[int | None, int | None]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, None))
                continue

            vaddr, vdata0 = lane_inputs[lane]
            old_val, new_mem = self.lds.atomic_rsub_rtn_u32(
                vaddr=vaddr,
                vdata0=vdata0,
                offset=inst.offset.offset16,
            )

            self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdst.index, old_val)
            lane_results.append((old_val, new_mem))

        return lane_results


class TestLdsRsubRtnSIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, lds_size_bytes=4096)
        # Pre-seed each address with 10
        for lane in range(32):
            simd.lds.write_dword(lane * 4, 10)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)  # vaddr
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 100 + lane)  # vdata0

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            # old_val was 10
            self.assertEqual(results[lane][0], 10)
            # new_val is (100 + lane) - 10 = 90 + lane
            self.assertEqual(results[lane][1], 90 + lane)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 10)
            self.assertEqual(simd.lds.read_dword(lane * 4), 90 + lane)

    def test_wave64_convergent_shared_address(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=64, lds_size_bytes=4096)
        simd.lds.write_dword(0, 0)

        # Each lane i sets vdata0 = i + 1, all targeting address 0
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane + 1)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 64)

        # Re-compute expected sequence manually
        curr_mem = 0
        for lane in range(64):
            vdata0 = lane + 1
            expected_old = curr_mem
            expected_new = (vdata0 - expected_old) & MASK32
            self.assertEqual(results[lane], (expected_old, expected_new))
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected_old)
            curr_mem = expected_new

        self.assertEqual(simd.lds.read_dword(0), curr_mem)

    def test_wave32_exec_masking_partial_lanes(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, lds_size_bytes=4096)
        simd.set_exec_mask(0x000000FF)  # Only lanes 0..7 active
        canary = 0x12345678

        for lane in range(32):
            simd.lds.write_dword(lane * 4, 10)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)  # vdst canary
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)  # vaddr
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 50)  # vdata0

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)

        for lane in range(32):
            if lane < 8:
                self.assertEqual(results[lane], (10, 40))
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 10)
                self.assertEqual(simd.lds.read_dword(lane * 4), 40)
            else:
                self.assertEqual(results[lane], (None, None))
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary)
                self.assertEqual(simd.lds.read_dword(lane * 4), 10)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=64, lds_size_bytes=4096)
        simd.set_exec_mask(0)  # All inactive
        simd.lds.write_dword(0, 555)

        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 777)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 100)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)
        self.assertEqual(results, [(None, None)] * 64)
        self.assertEqual(simd.lds.read_dword(0), 555)
        for lane in range(64):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 777)


class TestLdsRsubRtnRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(lds_size_bytes=4096)

    def test_aliasing_vdst_equals_vaddr(self) -> None:
        # vdst = v0, vaddr = v0, vdata0 = v1
        self.interp.lds.write_dword(0x20, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x20)  # vaddr and vdst
        self.interp.write_reg_u32(RegKind.VGPR, 1, 150)   # vdata0

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 0),
            vdata0=Register(RegKind.VGPR, 1),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 50)
        # v0 receives pre-op value 100
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 100)
        # LDS at 0x20 is updated to 150 - 100 = 50
        self.assertEqual(self.interp.lds.read_dword(0x20), 50)

    def test_aliasing_vdst_equals_vdata0(self) -> None:
        # vdst = v0, vaddr = v1, vdata0 = v0
        self.interp.lds.write_dword(0x30, 40)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 100)   # vdata0 and vdst
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x30)  # vaddr

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 0),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 40)
        self.assertEqual(new_mem, 60)
        # v0 receives pre-op value 40
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 40)
        # LDS at 0x30 is updated to 100 - 40 = 60
        self.assertEqual(self.interp.lds.read_dword(0x30), 60)

    def test_aliasing_all_three_operands_same_register(self) -> None:
        # vdst = v0, vaddr = v0, vdata0 = v0 (v0 = 0)
        self.interp.lds.write_dword(0x00, 5)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x00)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 0),
            vdata0=Register(RegKind.VGPR, 0),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 5)
        self.assertEqual(new_mem, u32(0 - 5))
        # v0 receives pre-op value 5
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 5)
        # LDS at 0x00 updated to 0 - 5 = 0xFFFFFFFB
        self.assertEqual(self.interp.lds.read_dword(0x00), u32(-5))

    def test_unrelated_vgprs_preserved(self) -> None:
        for idx in range(10, 20):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0x88000000 + idx)

        self.interp.lds.write_dword(0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 50)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        self.interp.step(inst)

        for idx in range(10, 20):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), 0x88000000 + idx)

    def test_simd_aliasing_vdst_equals_vdata0(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, lds_size_bytes=4096)
        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 0),
        )
        for lane in range(32):
            simd.lds.write_dword(lane * 4, 100 * (lane + 1))
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 1000)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)

        results = simd.step(inst)
        for lane in range(32):
            expected_old = 100 * (lane + 1)
            expected_new = (1000 - expected_old) & MASK32
            self.assertEqual(results[lane], (expected_old, expected_new))
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected_old)
            self.assertEqual(simd.lds.read_dword(lane * 4), expected_new)


class TestLdsRsubRtnConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(lds_size_bytes=4096)

    def test_condition_codes_zero_pos_neg(self) -> None:
        # Zero condition: vdata0 = 50, memory = 50 -> new_mem = 0
        self.interp.lds.write_dword(0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 50)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)

        # Positive condition: vdata0 = 100, memory = 40 -> new_mem = 60
        self.interp.lds.write_dword(0, 40)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)
        self.interp.step(inst)
        cc_pos = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)

        # Negative condition: vdata0 = 40, memory = 100 -> new_mem = -60 (0xFFFFFFC4)
        self.interp.lds.write_dword(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 40)
        self.interp.step(inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)

    def test_borrow_wrapped_flag(self) -> None:
        # Borrow condition: vdata0 < memory -> 40 < 100
        self.interp.lds.write_dword(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 40)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.borrow_wrapped)

        # No borrow condition: vdata0 >= memory -> 100 >= 40
        self.interp.lds.write_dword(0, 40)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)
        self.interp.step(inst)
        cc_noborrow = self.interp.last_condition_codes
        self.assertIsNotNone(cc_noborrow)
        assert cc_noborrow is not None
        self.assertFalse(cc_noborrow.borrow_wrapped)

    def test_status_flags_address_and_bank(self) -> None:
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x80)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 10)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset16=0x14),
        )
        self.interp.step(inst)
        flags = self.interp.last_status_flags
        self.assertIsNotNone(flags)
        assert flags is not None
        # eff_addr = 0x80 + 0x14 = 0x94 (148)
        self.assertEqual(flags.effective_address, 0x94)
        # dword_index = 0x94 >> 2 = 37
        self.assertEqual(flags.dword_index, 37)
        # bank_index = 37 % 32 = 5
        self.assertEqual(flags.bank_index, 5)

    def test_carry_borrow_vcc_isolation(self) -> None:
        canary = 0xDEADBEEFCAFEBABE
        self.interp.vcc = canary

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, canary)


class TestLdsRsubRtnScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(lds_size_bytes=4096)

    def test_scalar_execution_basic(self) -> None:
        # Memory at 0x10 holds 20. vaddr=0x10, vdata0=100.
        self.interp.lds.write_dword(0x10, 20)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)  # vaddr
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)   # vdata0

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 20)
        self.assertEqual(new_mem, 80)
        # vdst receives old_val
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 20)
        # LDS memory updated to 80
        self.assertEqual(self.interp.lds.read_dword(0x10), 80)

    def test_scalar_execution_with_offset(self) -> None:
        # Memory at 0x24 holds 50. vaddr=0x20, offset=4, vdata0=200.
        self.interp.lds.write_dword(0x24, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x20)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 200)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset16=4),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_mem, 150)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 50)
        self.assertEqual(self.interp.lds.read_dword(0x24), 150)

    def test_scalar_chained_execution(self) -> None:
        self.interp.lds.write_dword(0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 200)

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 4),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 3),
        ))

        results = self.interp.run_program(prog)
        # Step 1: 100 - 10 = 90 (old=10)
        self.assertEqual(results[0], (10, 90))
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 10)
        # Step 2: 200 - 90 = 110 (old=90)
        self.assertEqual(results[1], (90, 110))
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 4), 90)
        self.assertEqual(self.interp.lds.read_dword(0), 110)

    def test_scalar_underflow_wrap(self) -> None:
        self.interp.lds.write_dword(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 20)

        inst = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        old_val, new_mem = self.interp.step(inst)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, u32(20 - 100))
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 100)
        self.assertEqual(self.interp.lds.read_dword(0), u32(20 - 100))


class TestLdsRsubRtnAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v1 = Register(RegKind.VGPR, 1)
        v2 = Register(RegKind.VGPR, 2)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v1.format_asm(), "v1")
        self.assertEqual(v2.format_asm(), "v2")

    def test_instruction_formatting(self) -> None:
        inst_no_off = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst_no_off.to_assembly(), "ds_rsub_rtn_u32 v0, v1, v2")

        inst_with_off = Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 6),
            vdata0=Register(RegKind.VGPR, 7),
            offset=DSOffset(offset16=128),
        )
        self.assertEqual(inst_with_off.to_assembly(), "ds_rsub_rtn_u32 v5, v6, v7 offset:128")

    def test_instruction_validation(self) -> None:
        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_RTN_U32,
                vdst=Register(RegKind.SGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                vdata0=Register(RegKind.VGPR, 2),
            )

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_RTN_U32,
                vdst=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.SGPR, 1),
                vdata0=Register(RegKind.VGPR, 2),
            )

        # vdata0 cannot be a multi-register pair
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_RSUB_RTN_U32,
                vdst=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                vdata0=Register(RegKind.VGPR, 2, count=2),
            )

    def test_offset_validation(self) -> None:
        with self.assertRaises(ValueError):
            DSOffset(offset16=-1)
        with self.assertRaises(ValueError):
            DSOffset(offset16=65536)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset16=4),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_RSUB_RTN_U32,
            vdst=Register(RegKind.VGPR, 3),
            vaddr=Register(RegKind.VGPR, 1),
            vdata0=Register(RegKind.VGPR, 4),
        ))
        expected = "ds_rsub_rtn_u32 v0, v1, v2 offset:4\nds_rsub_rtn_u32 v3, v1, v4"
        self.assertEqual(prog.to_assembly(), expected)


class TestLDSMemoryAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS memory model."""

    def test_oracle_basic_rsub(self) -> None:
        # 50 - 10 = 40
        old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=10, vdata0=50)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_mem, 40)

        # Reverse order: 10 - 50 = -40 = 0xFFFFFFD8
        old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=50, vdata0=10)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_mem, u32(-40))
        self.assertEqual(new_mem, 0xFFFFFFD8)

    def test_oracle_identity(self) -> None:
        # X - X = 0
        for x in (0, 1, 42, 0x12345678, 0xFFFFFFFF):
            old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=x, vdata0=x)
            self.assertEqual(old_val, x & MASK32)
            self.assertEqual(new_mem, 0)

    def test_oracle_zero_cases(self) -> None:
        # 0 - 0 = 0
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 0), (0, 0))
        # vdata0 - 0 = vdata0
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 500), (0, 500))
        # 0 - current_mem = -current_mem
        self.assertEqual(ref_ds_rsub_rtn_u32(500, 0), (500, u32(-500)))

    def test_oracle_wraparound(self) -> None:
        # 0 - 0xFFFFFFFF = 1
        self.assertEqual(ref_ds_rsub_rtn_u32(0xFFFFFFFF, 0), (0xFFFFFFFF, 1))
        # 0xFFFFFFFF - 0 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 0xFFFFFFFF), (0, 0xFFFFFFFF))
        # 1 - 2 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(2, 1), (2, 0xFFFFFFFF))

    def test_lds_memory_initialization(self) -> None:
        mem = LDSMemory(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_dwords, 16384)
        self.assertEqual(len(mem.dwords), 16384)

        # Invalid non-power-of-two or non-positive sizes
        for invalid_size in (0, -64, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory(invalid_size)

    def test_lds_memory_read_write(self) -> None:
        mem = LDSMemory(1024)  # 256 dwords
        mem.write_dword(0x00, 0x12345678)
        self.assertEqual(mem.read_dword(0x00), 0x12345678)

        mem.write_dword(0x40, 0xABCDEF01)
        self.assertEqual(mem.read_dword(0x40), 0xABCDEF01)

    def test_lds_memory_unaligned_byte_address(self) -> None:
        mem = LDSMemory(1024)
        # Low 2 bits dropped: 0x100, 0x101, 0x102, 0x103 all map to dword index 0x100 >> 2
        mem.write_dword(0x100, 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x101), 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x102), 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x103), 0xDEADBEEF)

    def test_lds_memory_address_wrapping(self) -> None:
        mem = LDSMemory(1024)  # 256 dwords (1024 bytes)
        # Byte address 1024 wraps to 0
        mem.write_dword(0, 0xCAFE)
        self.assertEqual(mem.read_dword(1024), 0xCAFE)
        self.assertEqual(mem.read_dword(2048), 0xCAFE)

    def test_lds_memory_bank_calculation(self) -> None:
        # Bank is (eff_addr >> 2) % 32
        self.assertEqual(calc_lds_bank(0), 0)
        self.assertEqual(calc_lds_bank(4), 1)
        self.assertEqual(calc_lds_bank(124), 31)
        self.assertEqual(calc_lds_bank(128), 0)  # Next bank cycle

    def test_lds_atomic_rsub_rtn_u32(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x20, 100)

        # Execute rsub with vdata0=250: old=100, new = 250 - 100 = 150
        old_val, new_mem = mem.atomic_rsub_rtn_u32(vaddr=0x20, vdata0=250)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 150)
        self.assertEqual(mem.read_dword(0x20), 150)

        # Subsequent rsub with vdata0=50: old=150, new = 50 - 150 = -100 = 0xFFFFFF9C
        old_val2, new_mem2 = mem.atomic_rsub_rtn_u32(vaddr=0x20, vdata0=50)
        self.assertEqual(old_val2, 150)
        self.assertEqual(new_mem2, u32(-100))
        self.assertEqual(mem.read_dword(0x20), u32(-100))

    def test_lds_atomic_rsub_rtn_with_offset(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x50, 40)

        # vaddr = 0x40, offset = 0x10 -> eff_addr = 0x50
        old_val, new_mem = mem.atomic_rsub_rtn_u32(vaddr=0x40, vdata0=100, offset=0x10)
        self.assertEqual(old_val, 40)
        self.assertEqual(new_mem, 60)
        self.assertEqual(mem.read_dword(0x50), 60)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_RSUB_RTN_U32 = "ds_rsub_rtn_u32"


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
        limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < limit):
            raise ValueError(f"Register index out of range: {self.reg_type.value}{self.index}")

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class Immediate:
    """32-bit unsigned immediate value."""
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u32(self.value))

    @property
    def name(self) -> str:
        return hex(self.value)


@dataclass(frozen=True)
class OffsetImmediate:
    """16-bit immediate byte offset descriptor."""
    offset: int

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"Offset out of 16-bit range: {self.offset}")


@dataclass(frozen=True)
class DSRSubRtnInstruction:
    """Symbolic AST node representing a ds_rsub_rtn_u32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: Union[RegisterOperand, Immediate]
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_RSUB_RTN_U32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be VGPR, got {self.vdst.reg_type.value}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be VGPR, got {self.vaddr.reg_type.value}")
        if isinstance(self.vdata0, RegisterOperand) and self.vdata0.reg_type != RegType.VGPR:
            raise ValueError(f"vdata0 register must be VGPR, got {self.vdata0.reg_type.value}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of 16-bit range: {self.offset}")

    def disasm(self) -> str:
        vdata_str = self.vdata0.name
        parts = [f"{self.opcode.value} {self.vdst.name}, {self.vaddr.name}, {vdata_str}"]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)

    @classmethod
    def parse(cls, line: str) -> DSRSubRtnInstruction:
        raw = line.split("//")[0].split(";")[0].strip()
        if not raw:
            raise ValueError("Empty instruction line")
        tokens = raw.replace(",", " ").split()
        if not tokens:
            raise ValueError("No tokens found")
        mnemonic = tokens[0].lower()
        if mnemonic != DSOpcode.DS_RSUB_RTN_U32.value:
            raise ValueError(f"Unknown opcode: {mnemonic}")

        if len(tokens) < 4:
            raise ValueError(f"Insufficient operands for {mnemonic}")

        def parse_reg(tok: str) -> RegisterOperand:
            m = re.match(r"^([vs])\[?(\d+)\]?$", tok.lower())
            if not m:
                raise ValueError(f"Invalid register operand: {tok}")
            rtype = RegType.VGPR if m.group(1) == "v" else RegType.SGPR
            idx = int(m.group(2))
            return RegisterOperand(idx, rtype)

        vdst = parse_reg(tokens[1])
        vaddr = parse_reg(tokens[2])
        vdata_tok = tokens[3]
        if vdata_tok.lower().startswith("v") or vdata_tok.lower().startswith("s"):
            vdata0: Union[RegisterOperand, Immediate] = parse_reg(vdata_tok)
        else:
            vdata0 = Immediate(int(vdata_tok, 0))

        offset = 0
        gds = False
        for opt in tokens[4:]:
            opt_l = opt.lower()
            if opt_l == "gds":
                gds = True
            elif opt_l.startswith("offset:"):
                offset = int(opt_l.split(":", 1)[1], 0)
            else:
                raise ValueError(f"Unexpected token in instruction: {opt}")

        return cls(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=offset, gds=gds)


class TestDSRSubRtnAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_and_disasm(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v1 = RegisterOperand(1, RegType.VGPR)
        v2 = RegisterOperand(2, RegType.VGPR)

        inst_simple = DSRSubRtnInstruction(vdst=v0, vaddr=v1, vdata0=v2)
        self.assertEqual(inst_simple.disasm(), "ds_rsub_rtn_u32 v0, v1, v2")
        self.assertEqual(inst_simple.offset, 0)
        self.assertFalse(inst_simple.gds)

        inst_opts = DSRSubRtnInstruction(vdst=v0, vaddr=v1, vdata0=v2, offset=32, gds=True)
        self.assertEqual(inst_opts.disasm(), "ds_rsub_rtn_u32 v0, v1, v2 offset:32 gds")

        inst_imm = DSRSubRtnInstruction(vdst=v0, vaddr=v1, vdata0=Immediate(0x100), offset=16)
        self.assertEqual(inst_imm.disasm(), "ds_rsub_rtn_u32 v0, v1, 0x100 offset:16")

    def test_ast_parse_canonical(self) -> None:
        inst1 = DSRSubRtnInstruction.parse("ds_rsub_rtn_u32 v0, v1, v2")
        self.assertEqual(inst1.vdst, RegisterOperand(0, RegType.VGPR))
        self.assertEqual(inst1.vaddr, RegisterOperand(1, RegType.VGPR))
        self.assertEqual(inst1.vdata0, RegisterOperand(2, RegType.VGPR))
        self.assertEqual(inst1.offset, 0)
        self.assertFalse(inst1.gds)

        inst2 = DSRSubRtnInstruction.parse("ds_rsub_rtn_u32 v[5], v[6], v[7] offset:64 gds")
        self.assertEqual(inst2.vdst, RegisterOperand(5, RegType.VGPR))
        self.assertEqual(inst2.vaddr, RegisterOperand(6, RegType.VGPR))
        self.assertEqual(inst2.vdata0, RegisterOperand(7, RegType.VGPR))
        self.assertEqual(inst2.offset, 64)
        self.assertTrue(inst2.gds)

        inst3 = DSRSubRtnInstruction.parse("ds_rsub_rtn_u32 v10, v11, 0x200 offset:0x10")
        self.assertEqual(inst3.vdata0, Immediate(0x200))
        self.assertEqual(inst3.offset, 16)

    def test_ast_validation_errors(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=s0, vaddr=v0, vdata0=v0)

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=s0, vdata0=v0)

        # vdata0 cannot be SGPR
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=v0, vdata0=s0)

        # Negative offset
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=-1)

        # Offset exceeding 16 bits
        with self.assertRaises(ValueError):
            DSRSubRtnInstruction(vdst=v0, vaddr=v0, vdata0=v0, offset=65536)

        # Register index out of bounds
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState:
    """Architectural state for a single execution thread with scalar register file and LDS."""

    def __init__(
        self,
        lds: Optional[LDSMemory] = None,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
        num_sgprs: int = DEFAULT_NUM_SGPRS,
    ) -> None:
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.vgprs: List[int] = [0] * num_vgprs
        self.sgprs: List[int] = [0] * num_sgprs

    def get_vgpr(self, reg: Union[RegisterOperand, int]) -> int:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        return self.vgprs[idx]

    def set_vgpr(self, reg: Union[RegisterOperand, int], val: int) -> None:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        self.vgprs[idx] = u32(val)

    def get_sgpr(self, reg: Union[RegisterOperand, int]) -> int:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        return self.sgprs[idx]

    def set_sgpr(self, reg: Union[RegisterOperand, int], val: int) -> None:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        self.sgprs[idx] = u32(val)


class ScalarDSInterpreter:
    """Deterministic scalar interpreter executing DS instructions."""

    def execute(self, state: ScalarThreadState, inst: DSRSubRtnInstruction) -> Tuple[int, int]:
        # 1. Latch input operands before any register write
        base_addr = state.get_vgpr(inst.vaddr)
        eff_addr = calc_lds_effective_address(base_addr, inst.offset)

        if isinstance(inst.vdata0, RegisterOperand):
            vdata_val = state.get_vgpr(inst.vdata0)
        else:
            vdata_val = inst.vdata0.value

        # 2. Atomic memory transaction on LDS
        old_val, new_mem = state.lds.atomic_rsub_rtn_u32(
            vaddr=base_addr,
            vdata0=vdata_val,
            offset=inst.offset,
        )

        # 3. Write return value to destination VGPR
        state.set_vgpr(inst.vdst, old_val)

        return old_val, new_mem


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_execution_basic(self) -> None:
        # Initial memory at 0x40 is 1000
        addr = 0x40
        self.lds.write_dword(addr, 1000)

        # v1 has address 0x40, v2 has vdata0 2500, v0 is vdst
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 2500)
        self.state.set_vgpr(0, 0xCAFEBABE)

        inst = DSRSubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        # 2500 - 1000 = 1500
        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, 1000)
        self.assertEqual(new_mem, 1500)
        self.assertEqual(self.state.get_vgpr(0), 1000)  # vdst received old_val
        self.assertEqual(self.lds.read_dword(addr), 1500)  # LDS received new_mem

    def test_scalar_execution_with_offset(self) -> None:
        base_addr = 0x100
        offset = 0x14
        eff_addr = base_addr + offset

        self.lds.write_dword(eff_addr, 42)
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, 100)

        inst = DSRSubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=offset,
        )

        # 100 - 42 = 58
        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, 42)
        self.assertEqual(new_mem, 58)
        self.assertEqual(self.state.get_vgpr(0), 42)
        self.assertEqual(self.lds.read_dword(eff_addr), 58)

    def test_scalar_execution_with_immediate(self) -> None:
        addr = 0x200
        self.lds.write_dword(addr, 0x200)
        self.state.set_vgpr(1, addr)

        inst = DSRSubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=Immediate(0x500),
        )

        # 0x500 - 0x200 = 0x300
        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, 0x200)
        self.assertEqual(new_mem, 0x300)
        self.assertEqual(self.state.get_vgpr(0), 0x200)
        self.assertEqual(self.lds.read_dword(addr), 0x300)

    def test_scalar_unaligned_byte_address(self) -> None:
        # 0x303 maps to 0x300
        self.lds.write_dword(0x300, 70)
        self.state.set_vgpr(1, 0x303)
        self.state.set_vgpr(2, 100)

        inst = DSRSubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, 70)
        self.assertEqual(new_mem, 30)
        self.assertEqual(self.lds.read_dword(0x300), 30)

    def test_scalar_reverse_sub_wraparound(self) -> None:
        addr = 0x80
        self.lds.write_dword(addr, 100)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 30)

        inst = DSRSubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        # 30 - 100 = -70 = 0xFFFFFFBA
        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, u32(-70))
        self.assertEqual(self.state.get_vgpr(0), 100)
        self.assertEqual(self.lds.read_dword(addr), 0xFFFFFFBA)


class TestLdsRsubRtnBoundaryExtremes(unittest.TestCase):
    """Unit tests for boundary extremes, address wrapping, and IEEE 754 bit pattern handling."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def test_integer_extremes_and_overflow(self) -> None:
        # 0 - 0 = 0
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 0), (0, 0))
        # 0 - 1 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(1, 0), (1, 0xFFFFFFFF))
        # 0 - 0xFFFFFFFF = 1
        self.assertEqual(ref_ds_rsub_rtn_u32(0xFFFFFFFF, 0), (0xFFFFFFFF, 1))
        # 0x7FFFFFFF - 0x80000000 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(0x80000000, 0x7FFFFFFF), (0x80000000, 0xFFFFFFFF))
        # 0x80000000 - 0x7FFFFFFF = 1
        self.assertEqual(ref_ds_rsub_rtn_u32(0x7FFFFFFF, 0x80000000), (0x7FFFFFFF, 1))
        # 0xFFFFFFFF - 0xFFFFFFFF = 0
        self.assertEqual(ref_ds_rsub_rtn_u32(0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0))

    def test_float_and_nan_bit_patterns_preserve_integer_arithmetic(self) -> None:
        # ds_rsub_rtn_u32 operates purely bitwise, preserving float/NaN bit patterns
        patterns = [
            0x7FC00000,  # Single-precision quiet NaN
            0x7F800001,  # Single-precision signaling NaN
            0x7F800000,  # Single-precision +Infinity
            0xFF800000,  # Single-precision -Infinity
            0x00000001,  # Subnormal / Denormal float
            0x80000000,  # Negative zero (-0.0)
        ]
        addr = 0
        vdata0 = 0x12345678
        for pat in patterns:
            self.lds.write_dword(addr, pat)
            old_val, new_mem = self.lds.atomic_rsub_rtn_u32(vaddr=addr, vdata0=vdata0)
            self.assertEqual(old_val, pat)
            expected_new = (vdata0 - pat) & MASK32
            self.assertEqual(new_mem, expected_new)
            self.assertEqual(self.lds.read_dword(addr), expected_new)
            addr += 4

    def test_address_boundary_extremes(self) -> None:
        # Max 16-bit offset (0xFFFF = 65535)
        eff_addr = calc_lds_effective_address(vaddr=0x1000, offset=0xFFFF)
        self.assertEqual(eff_addr, 0x1000 + 65535)

        # 32-bit address wrapping
        eff_wrap = calc_lds_effective_address(vaddr=0xFFFFFFFF, offset=1)
        self.assertEqual(eff_wrap, 0)

        # Max dword index wraps within LDS capacity
        idx = calc_lds_dword_index(eff_addr=0xFFFFFFFC, capacity_dwords=16384)
        self.assertEqual(idx, 16383)


if __name__ == "__main__":
    unittest.main()
