"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_sub_x2.

Models AMD RDNA2 / GFX10 MUBUF 64-bit untyped buffer memory atomic subtraction:
  buffer_atomic_sub_x2 vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR pair v[d:d+1] containing the 64-bit value to atomically subtract from memory.
       If 'glc' is enabled, the 64-bit pre-operation value at memory is returned to v[d:d+1].
       If 'glc' is omitted, vdata registers retain their input values.
     - vaddr: VGPR containing byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = (soffset & 0xFFFFFFFF) + (inst_offset & 0xFFF) + (vaddr & 0xFFFFFFFF if offen else 0)
     If effective_offset + 8 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic operation is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to v[d:d+1].
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory_64[effective_address]
       new_val = (old_val - (vdata & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
       memory_64[effective_address] = new_val
       if glc: vdata = old_val
  4. Inactive Lanes (EXEC mask):
     Lanes disabled by EXEC mask never touch memory and do not modify their VGPRs.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned integer as signed two's-complement."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_BIT64) else val


@dataclasses.dataclass(frozen=True)
class BufferResourceDescriptor:
    """AMD GFX10 128-bit Buffer Resource Descriptor (V#)."""
    base_address: int
    stride: int
    num_records: int
    format_flags: int = 0

    @classmethod
    def from_dwords(cls, d0: int, d1: int, d2: int, d3: int) -> BufferResourceDescriptor:
        base_lo = d0 & MASK32
        base_hi = d1 & 0xFFFF
        base = (base_hi << 32) | base_lo
        stride = (d1 >> 16) & 0xFFFF
        num_records = d2 & MASK32
        return cls(base_address=base, stride=stride, num_records=num_records, format_flags=d3)

    def to_dwords(self) -> tuple[int, int, int, int]:
        d0 = self.base_address & MASK32
        d1 = ((self.base_address >> 32) & 0xFFFF) | ((self.stride & 0xFFFF) << 16)
        d2 = self.num_records & MASK32
        d3 = self.format_flags & MASK32
        return d0, d1, d2, d3


class BufferMemory:
    """Byte-addressable memory buffer simulating GPU global / buffer device memory."""

    def __init__(self, size: int = 65536) -> None:
        self.data: bytearray = bytearray(size)

    def read_u32(self, addr: int) -> int:
        if addr < 0 or addr + 4 > len(self.data):
            return 0
        return struct.unpack_from("<I", self.data, addr)[0]

    def write_u32(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 4 > len(self.data):
            return
        struct.pack_into("<I", self.data, addr, val & MASK32)

    def read_u64(self, addr: int) -> int:
        if addr < 0 or addr + 8 > len(self.data):
            return 0
        return struct.unpack_from("<Q", self.data, addr)[0]

    def write_u64(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 8 > len(self.data):
            return
        struct.pack_into("<Q", self.data, addr, val & MASK64)


def ref_buffer_atomic_sub_x2(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    sub_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_sub_x2.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS (64-bit access requires 8 bytes)
    if effective_offset + 8 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc else sub_val
        return (0 if glc else (sub_val & MASK64)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u64(phys_addr)
    new_val = (old_val - (sub_val & MASK64)) & MASK64
    mem.write_u64(phys_addr, new_val)

    ret_val = old_val if glc else (sub_val & MASK64)
    return ret_val, True


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
class BufferAtomicModifiers:
    """Modifiers and options for buffer atomic instructions."""
    offset: int = 0
    glc: bool = False
    slc: bool = False
    offen: bool = True
    idxen: bool = False

    def __post_init__(self) -> None:
        if self.offset < 0 or self.offset > 4095:
            raise ValueError(f"Immediate inst_offset must be in range [0, 4095]: {self.offset}")


class Opcode(enum.Enum):
    """Buffer atomic opcodes."""
    BUFFER_ATOMIC_SUB_X2 = "buffer_atomic_sub_x2"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_sub_x2 instruction."""
    opcode: Opcode
    vdata: Register
    vaddr: Register
    srsrc: Register
    soffset: Register | int
    mods: BufferAtomicModifiers = dataclasses.field(default_factory=BufferAtomicModifiers)

    def __post_init__(self) -> None:
        if not self.vdata.is_vgpr or self.vdata.count != 2:
            raise ValueError(f"vdata must be a 64-bit VGPR pair (count=2): {self.vdata}")
        if not self.vaddr.is_vgpr:
            raise ValueError(f"vaddr must be a VGPR: {self.vaddr}")
        if not self.srsrc.is_sgpr or self.srsrc.count != 4:
            raise ValueError(f"srsrc must be a 128-bit SGPR group (count=4): {self.srsrc}")
        if isinstance(self.soffset, Register) and not self.soffset.is_sgpr:
            raise ValueError(f"soffset register must be an SGPR: {self.soffset}")

    def to_assembly(self) -> str:
        vd_str = self.vdata.format_asm()
        va_str = self.vaddr.format_asm()
        sr_str = self.srsrc.format_asm()
        so_str = self.soffset.format_asm() if isinstance(self.soffset, Register) else str(self.soffset)

        mod_parts: list[str] = []
        if self.mods.offen:
            mod_parts.append("offen")
        if self.mods.idxen:
            mod_parts.append("idxen")
        if self.mods.offset != 0:
            mod_parts.append(f"offset:{self.mods.offset}")
        if self.mods.glc:
            mod_parts.append("glc")
        if self.mods.slc:
            mod_parts.append("slc")

        mods_str = (" " + " ".join(mod_parts)) if mod_parts else ""
        return f"{self.opcode.value} {vd_str}, {va_str}, {sr_str}, {so_str}{mods_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of buffer instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferAtomicSubX2ConditionCodes:
    """Architectural integer condition codes for 64-bit buffer atomic operations."""
    zero: bool
    negative: bool
    positive: bool
    borrow: bool


@dataclasses.dataclass(frozen=True)
class BufferStatusFlags:
    """Buffer memory status flags."""
    in_bounds: bool
    oob_fault: bool


def compute_buffer_sub_x2_condition_codes(old_val: int, sub_val: int, ret_val: int) -> BufferAtomicSubX2ConditionCodes:
    """Computes condition codes evaluating 64-bit result and subtraction borrow."""
    signed_ret = s64(ret_val)
    borrow = (old_val & MASK64) < (sub_val & MASK64)
    return BufferAtomicSubX2ConditionCodes(
        zero=(signed_ret == 0),
        negative=(signed_ret < 0),
        positive=(signed_ret > 0),
        borrow=borrow,
    )


def compute_buffer_status_flags(in_bounds: bool) -> BufferStatusFlags:
    """Computes buffer memory bounds flags."""
    return BufferStatusFlags(
        in_bounds=in_bounds,
        oob_fault=not in_bounds,
    )


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_sub_x2."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.vcc: int = 0
        self.last_condition_codes: BufferAtomicSubX2ConditionCodes | None = None
        self.last_status_flags: BufferStatusFlags | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_reg_u64(self, idx: int) -> int:
        lo = self.read_reg_u32(RegKind.VGPR, idx)
        hi = self.read_reg_u32(RegKind.VGPR, idx + 1)
        return (hi << 32) | lo

    def write_reg_u64(self, idx: int, val: int) -> None:
        self.write_reg_u32(RegKind.VGPR, idx, val & MASK32)
        self.write_reg_u32(RegKind.VGPR, idx + 1, (val >> 32) & MASK32)

    def set_buffer_descriptor(self, idx: int, desc: BufferResourceDescriptor) -> None:
        d0, d1, d2, d3 = desc.to_dwords()
        self.write_reg_u32(RegKind.SGPR, idx, d0)
        self.write_reg_u32(RegKind.SGPR, idx + 1, d1)
        self.write_reg_u32(RegKind.SGPR, idx + 2, d2)
        self.write_reg_u32(RegKind.SGPR, idx + 3, d3)

    def get_buffer_descriptor(self, idx: int) -> BufferResourceDescriptor:
        d0 = self.read_reg_u32(RegKind.SGPR, idx)
        d1 = self.read_reg_u32(RegKind.SGPR, idx + 1)
        d2 = self.read_reg_u32(RegKind.SGPR, idx + 2)
        d3 = self.read_reg_u32(RegKind.SGPR, idx + 3)
        return BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

    def step(self, inst: Instruction) -> tuple[int, bool]:
        """Executes a single buffer_atomic_sub_x2 instruction."""
        vdata = self.read_reg_u64(inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        ret_val, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            sub_val=vdata,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u64(inst.vdata.index, ret_val)

        self.last_condition_codes = compute_buffer_sub_x2_condition_codes(
            ret_val if inst.mods.glc else 0, vdata, ret_val
        )
        self.last_status_flags = compute_buffer_status_flags(in_bounds)
        # Note: buffer atomic instructions never modify VCC; carry/borrow register remains untouched.

        self.trace.append(f"{inst.to_assembly()} => ret={ret_val:#018x}, in_bounds={in_bounds}")
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDBufferWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for buffer atomics."""

    def __init__(self, wave_size: int = 64, mem_size: int = 65536) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)

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

    def read_lane_reg_u64(self, lane: int, idx: int) -> int:
        lo = self.read_lane_reg_u32(lane, RegKind.VGPR, idx)
        hi = self.read_lane_reg_u32(lane, RegKind.VGPR, idx + 1)
        return (hi << 32) | lo

    def write_lane_reg_u64(self, lane: int, idx: int, val: int) -> None:
        self.write_lane_reg_u32(lane, RegKind.VGPR, idx, val & MASK32)
        self.write_lane_reg_u32(lane, RegKind.VGPR, idx + 1, (val >> 32) & MASK32)

    def set_buffer_descriptor(self, idx: int, desc: BufferResourceDescriptor) -> None:
        d0, d1, d2, d3 = desc.to_dwords()
        self.write_lane_reg_u32(0, RegKind.SGPR, idx, d0)
        self.write_lane_reg_u32(0, RegKind.SGPR, idx + 1, d1)
        self.write_lane_reg_u32(0, RegKind.SGPR, idx + 2, d2)
        self.write_lane_reg_u32(0, RegKind.SGPR, idx + 3, d3)

    def get_buffer_descriptor(self, idx: int) -> BufferResourceDescriptor:
        d0 = self.read_lane_reg_u32(0, RegKind.SGPR, idx)
        d1 = self.read_lane_reg_u32(0, RegKind.SGPR, idx + 1)
        d2 = self.read_lane_reg_u32(0, RegKind.SGPR, idx + 2)
        d3 = self.read_lane_reg_u32(0, RegKind.SGPR, idx + 3)
        return BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

    def step(self, inst: Instruction) -> list[tuple[int | None, bool]]:
        """Executes a buffer_atomic_sub_x2 instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        # 1. Latch inputs for each lane
        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            vd = self.read_lane_reg_u64(lane, inst.vdata.index)
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((vd, va))

        # 2. Serialize atomic operations across active lanes
        lane_results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, False))
                continue

            vdata, vaddr = lane_inputs[lane]
            ret_val, in_bounds = ref_buffer_atomic_sub_x2(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr,
                sub_val=vdata,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if inst.mods.glc:
                self.write_lane_reg_u64(lane, inst.vdata.index, ret_val)

            lane_results.append((ret_val, in_bounds))

        return lane_results


class TestBufferAtomicSubX2SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=4096,
        )

    def test_wave32_divergent_addresses(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        for lane in range(32):
            simd.mem.write_u64(lane * 8, 1000 + lane)
            simd.write_lane_reg_u64(lane, 0, lane)                  # sub value
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane * 8) # address

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertTrue(results[lane][1])
            self.assertEqual(simd.mem.read_u64(lane * 8), 1000)

    def test_wave64_convergent_atomic_sub_x2(self) -> None:
        # All 64 lanes subtract 1 from address 0 (initially 64)
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.mem.write_u64(0, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        for lane in range(64):
            simd.write_lane_reg_u64(lane, 0, 1)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0)

        results = simd.step(inst)
        # Lane 0 received initial 64
        self.assertEqual(results[0][0], 64)
        # Final memory must be 0
        self.assertEqual(simd.mem.read_u64(0), 0)

    def test_wave32_exec_masking_skips_inactive(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        # Active mask: lanes 0..3 active (0x0F)
        simd.set_exec_mask(0x0000000F)

        for lane in range(32):
            simd.mem.write_u64(lane * 8, 500)
            simd.write_lane_reg_u64(lane, 0, 100)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane * 8)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        # Active lanes: memory reduced to 400, glc returns 500
        for lane in range(4):
            self.assertEqual(simd.mem.read_u64(lane * 8), 400)
            self.assertEqual(results[lane], (500, True))
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 500)

        # Inactive lanes: memory untouched (500), result (None, False), register untouched (100)
        for lane in range(4, 32):
            self.assertEqual(simd.mem.read_u64(lane * 8), 500)
            self.assertEqual(results[lane], (None, False))
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 100)

    def test_wave64_alternating_exec_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        # Even lanes active (0x5555555555555555)
        simd.set_exec_mask(0x5555555555555555)

        for lane in range(64):
            simd.mem.write_u64(lane * 8, 1000)
            simd.write_lane_reg_u64(lane, 0, 250)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane * 8)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        results = simd.step(inst)
        for lane in range(64):
            if lane % 2 == 0:
                self.assertTrue(results[lane][1])
                self.assertEqual(simd.mem.read_u64(lane * 8), 750)
            else:
                self.assertFalse(results[lane][1])
                self.assertEqual(simd.mem.read_u64(lane * 8), 1000)


class TestBufferAtomicSubX2RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_vdata_vaddr_aliasing(self) -> None:
        # v0 acts as vaddr (offset 64) and low 32 bits of vdata v[0:1]
        addr_and_val = 64
        self.interp.mem.write_u64(addr_and_val, 500)
        self.interp.write_reg_u64(0, addr_and_val)  # v0 = 64, v1 = 0

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # Pre-op memory value was 500
        self.assertEqual(ret, 500)
        # Memory at 64 becomes 500 - 64 = 436
        self.assertEqual(self.interp.mem.read_u64(addr_and_val), 436)
        # v[0:1] overwritten with pre-op value (500)
        self.assertEqual(self.interp.read_reg_u64(0), 500)

    def test_canary_neighbor_vgprs_unmodified(self) -> None:
        # Preload canary values into VGPRs 0..31
        canaries = [0xDEAD0000 + i for i in range(32)]
        for i, val in enumerate(canaries):
            self.interp.write_reg_u32(RegKind.VGPR, i, val)

        # Set address to 64 and value to 50
        self.interp.write_reg_u64(0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 64)
        self.interp.mem.write_u64(64, 200)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        self.interp.step(inst)

        # Registers 3..31 must strictly retain their canaries
        for idx in range(3, 32):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), canaries[idx])

    def test_sgpr_descriptor_and_offset_preserved(self) -> None:
        d0, d1, d2, d3 = self.desc.to_dwords()
        self.interp.write_reg_u32(RegKind.SGPR, 8, 16)  # soffset

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        # SGPR descriptor and soffset registers must be completely untouched
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 4), d0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 5), d1)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 6), d2)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 7), d3)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 8), 16)

    def test_simd_aliasing_with_divergent_exec_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x0000000F  # Lanes 0..3 active
        simd.set_exec_mask(exec_mask)
        simd.mem.write_u64(16, 1000)

        for lane in range(32):
            simd.write_lane_reg_u64(lane, 0, 16)               # v[0:1] = 16
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 16) # v0 = 16 (aliased)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Active lanes: v[0:1] updated to pre-op memory values in sequential order
        # Lane 0 sees 1000, updates mem to 1000 - 16 = 984.
        # Lane 1 sees 984, updates mem to 984 - 16 = 968.
        # Lane 2 sees 968, updates mem to 968 - 16 = 952.
        # Lane 3 sees 952, updates mem to 952 - 16 = 936.
        self.assertEqual(simd.read_lane_reg_u64(0, 0), 1000)
        self.assertEqual(simd.read_lane_reg_u64(1, 0), 984)
        self.assertEqual(simd.read_lane_reg_u64(2, 0), 968)
        self.assertEqual(simd.read_lane_reg_u64(3, 0), 952)
        # Inactive lanes: v[0:1] untouched (16)
        for lane in range(4, 32):
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 16)


class TestBufferAtomicSubX2ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_condition_codes_zero_pos_neg_borrow(self) -> None:
        # Pre-seed memory at 0 with 100
        self.interp.mem.write_u64(0, 100)
        self.interp.write_reg_u64(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        # Pre-op return value is 100 (positive)
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertFalse(cc.zero)
        self.assertFalse(cc.negative)
        self.assertTrue(cc.positive)
        self.assertFalse(cc.borrow)

        # Pre-seed memory with 0
        self.interp.mem.write_u64(0, 0)
        self.interp.write_reg_u64(0, 10)
        self.interp.step(inst)
        cc_zero = self.interp.last_condition_codes
        self.assertIsNotNone(cc_zero)
        assert cc_zero is not None
        self.assertTrue(cc_zero.zero)
        self.assertTrue(cc_zero.borrow)  # 0 < 10 -> borrow occurred

        # Pre-seed memory with negative two's complement value (0x8000000000000000)
        self.interp.mem.write_u64(0, 0x8000000000000000)
        self.interp.write_reg_u64(0, 1)
        self.interp.step(inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertTrue(cc_neg.negative)

    def test_status_flags_in_bounds_and_oob(self) -> None:
        self.interp.write_reg_u64(0, 1)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)  # in bounds (< 1024)

        inst_in = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst_in)
        flags_in = self.interp.last_status_flags
        self.assertIsNotNone(flags_in)
        assert flags_in is not None
        self.assertTrue(flags_in.in_bounds)
        self.assertFalse(flags_in.oob_fault)

        # OOB access (offset 2000 > 1024)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 2000)
        self.interp.step(inst_in)
        flags_oob = self.interp.last_status_flags
        self.assertIsNotNone(flags_oob)
        assert flags_oob is not None
        self.assertFalse(flags_oob.in_bounds)
        self.assertTrue(flags_oob.oob_fault)

    def test_carry_borrow_vcc_isolation(self) -> None:
        canary = 0x1234567887654321
        self.interp.vcc = canary

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        self.interp.write_reg_u64(0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)
        self.interp.step(inst)

        # VCC must not be modified by atomic operations
        self.assertEqual(self.interp.vcc, canary)


class TestBufferAtomicSubX2ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_sub_x2_execution(self) -> None:
        self.interp.mem.write_u64(64, 1000)
        self.interp.write_reg_u64(0, 400)               # v[0:1]
        self.interp.write_reg_u32(RegKind.VGPR, 2, 64)   # v2: vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(self.interp.mem.read_u64(64), 600)
        # Without glc, v[0:1] remains untouched
        self.assertEqual(self.interp.read_reg_u64(0), 400)

    def test_scalar_sub_x2_glc_writeback(self) -> None:
        self.interp.mem.write_u64(128, 0x2000000000000000)
        self.interp.write_reg_u64(0, 0x0500000000000000)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x2000000000000000)
        # v[0:1] updated to previous value
        self.assertEqual(self.interp.read_reg_u64(0), 0x2000000000000000)
        # Memory holds 0x2000000000000000 - 0x0500000000000000 = 0x1B00000000000000
        self.assertEqual(self.interp.mem.read_u64(128), 0x1B00000000000000)

    def test_scalar_oob_suppression(self) -> None:
        # Offset 1020 is out of bounds for 8-byte access
        self.interp.mem.write_u64(1020, 0x777788889999AAAA)
        self.interp.write_reg_u64(0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 1020)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.interp.read_reg_u64(0), 0)
        self.assertEqual(self.interp.mem.read_u64(1020), 0x777788889999AAAA)

    def test_scalar_program_chain(self) -> None:
        self.interp.mem.write_u64(0, 1000)
        self.interp.write_reg_u64(0, 200)
        self.interp.write_reg_u64(4, 300)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 4, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))

        self.interp.run_program(prog)
        # 1000 - 200 - 300 = 500
        self.assertEqual(self.interp.mem.read_u64(0), 500)
        self.assertEqual(len(self.interp.trace), 2)

    def test_soffset_from_sgpr(self) -> None:
        self.interp.mem.write_u64(100, 250)
        self.interp.write_reg_u64(0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 20)          # vaddr
        self.interp.write_reg_u32(RegKind.SGPR, 8, 80)          # soffset in s8

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 250)
        self.assertEqual(self.interp.mem.read_u64(100), 200)


class TestBufferAtomicSubX2AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=2)
        v2 = Register(RegKind.VGPR, 2, count=1)
        s_desc = Register(RegKind.SGPR, 4, count=4)
        s_off = Register(RegKind.SGPR, 8, count=1)

        self.assertEqual(v0.format_asm(), "v[0:1]")
        self.assertEqual(v2.format_asm(), "v2")
        self.assertEqual(s_desc.format_asm(), "s[4:7]")
        self.assertEqual(s_off.format_asm(), "s8")

    def test_instruction_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_sub_x2 v[0:1], v2, s[4:7], s8 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_sub_x2 v[0:1], v2, s[12:15], 0 offen",
        )

    def test_vdata_count_validation(self) -> None:
        # Invalid vdata count (must be 2 for _x2)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
                vdata=Register(RegKind.VGPR, 0, count=4),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_instruction_validation(self) -> None:
        # Invalid srsrc count (must be 4)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

        # Invalid offset (> 4095)
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=5000)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SUB_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        ))
        expected = "buffer_atomic_sub_x2 v[0:1], v2, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


class TestBufferAtomicSubX2Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_sub_x2_in_bounds(self) -> None:
        # Pre-seed address 64 with 64-bit value
        self.mem.write_u64(64, 1000)
        ret, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            sub_val=300,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u64(64), 700)

    def test_sub_x2_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u64(128, 0x1234567890ABCDEF)
        ret, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            sub_val=0x000000000000000F,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x1234567890ABCDEF)
        self.assertEqual(self.mem.read_u64(128), 0x1234567890ABCDE0)

    def test_sub_x2_oob_rejection(self) -> None:
        # num_records is 1024. Offset 1020 has 1020 + 8 = 1028 > 1024 (OOB)
        self.mem.write_u64(1020, 0x777788889999AAAA)
        ret, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1020,
            sub_val=10,
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(1020), 0x777788889999AAAA)

    def test_sub_x2_boundary_exact_fit(self) -> None:
        # Offset 1016 has 1016 + 8 == 1024 (exact fit, in-bounds)
        self.mem.write_u64(1016, 100)
        ret, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1016,
            sub_val=40,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 100)
        self.assertEqual(self.mem.read_u64(1016), 60)

    def test_sub_x2_wrapping_64bit_integer(self) -> None:
        self.mem.write_u64(0, 0)
        ret, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=0,
            sub_val=1,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(0), 0xFFFFFFFFFFFFFFFF)

    def test_sub_x2_negative_subtraction_two_complement(self) -> None:
        self.mem.write_u64(16, 100)
        ret, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=16,
            sub_val=s64(-50),
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 100)
        self.assertEqual(self.mem.read_u64(16), 150)

    def test_sub_x2_offset_combination(self) -> None:
        # Total offset = soffset(128) + inst_offset(64) + vaddr(8) = 200
        self.mem.write_u64(200, 500)
        ret, in_bounds = ref_buffer_atomic_sub_x2(
            self.mem,
            self.desc,
            soffset=128,
            vaddr=8,
            sub_val=125,
            inst_offset=64,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 500)
        self.assertEqual(self.mem.read_u64(200), 375)

    def test_descriptor_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x00007FFF12340000,
            stride=32,
            num_records=131072,
            format_flags=0x40,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc.base_address, restored.base_address)
        self.assertEqual(desc.stride, restored.stride)
        self.assertEqual(desc.num_records, restored.num_records)
        self.assertEqual(desc.format_flags, restored.format_flags)


if __name__ == "__main__":
    unittest.main()
