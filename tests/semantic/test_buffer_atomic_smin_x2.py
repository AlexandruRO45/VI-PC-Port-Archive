"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_smin_x2.

Models AMD RDNA2 / GFX10 MUBUF 64-bit untyped buffer memory atomic signed minimum:
  buffer_atomic_smin_x2 vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR pair v[d:d+1] containing the 64-bit signed integer value to atomically compare.
       If 'glc' (Globally Coherent) is enabled, the 64-bit pre-operation value at memory is returned to v[d:d+1].
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
       The atomic signed minimum is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to v[d:d+1].
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory_64[effective_address]
       new_val = min(s64(old_val), s64(vdata))
       memory_64[effective_address] = new_val & 0xFFFFFFFFFFFFFFFF
       if glc: vdata = old_val
  4. Inactive Lanes (EXEC mask):
     Lanes disabled by EXEC mask never touch memory and do not modify their VGPRs.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
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


def ref_buffer_atomic_smin_x2(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    smin_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_smin_x2.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS (64-bit access requires 8 bytes)
    if effective_offset + 8 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc else smin_val
        return (0 if glc else (smin_val & MASK64)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u64(phys_addr)
    new_val = min(s64(old_val), s64(smin_val)) & MASK64
    mem.write_u64(phys_addr, new_val)

    ret_val = old_val if glc else (smin_val & MASK64)
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
    BUFFER_ATOMIC_SMIN_X2 = "buffer_atomic_smin_x2"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_smin_x2 instruction."""
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
class BufferAtomicConditionCodes:
    """Architectural integer condition codes for 64-bit buffer atomic operations."""
    zero: bool
    negative: bool
    positive: bool
    updated: bool


@dataclasses.dataclass(frozen=True)
class BufferStatusFlags:
    """Buffer memory status flags."""
    in_bounds: bool
    oob_fault: bool


def compute_buffer_condition_codes(old_val: int, smin_val: int, ret_val: int) -> BufferAtomicConditionCodes:
    """Computes condition codes evaluating the result and minimum update status."""
    signed_ret = s64(ret_val)
    new_val = min(s64(old_val), s64(smin_val))
    return BufferAtomicConditionCodes(
        zero=(signed_ret == 0),
        negative=(signed_ret < 0),
        positive=(signed_ret > 0),
        updated=(new_val != s64(old_val)),
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
    """Deterministic single-lane scalar interpreter for 64-bit buffer instructions."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.vcc: int = 0
        self.last_condition_codes: BufferAtomicConditionCodes | None = None
        self.last_status_flags: BufferStatusFlags | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
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
        """Executes a single buffer_atomic_smin_x2 instruction deterministically."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        smin_val = self.read_reg_u64(inst.vdata.index)

        eff_off = (soffset & MASK32) + (inst.mods.offset & 0xFFF) + (vaddr & MASK32 if inst.mods.offen else 0)
        phys_addr = (desc.base_address + eff_off) & MASK64
        old_val = self.mem.read_u64(phys_addr) if eff_off + 8 <= desc.num_records else 0

        ret_val, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            smin_val=smin_val,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u64(inst.vdata.index, ret_val)

        self.last_condition_codes = compute_buffer_condition_codes(
            old_val if inst.mods.glc else ret_val, smin_val, ret_val
        )
        self.last_status_flags = compute_buffer_status_flags(in_bounds)

        self.trace.append(
            f"{inst.to_assembly()} => ret={ret_val:#018x}, in_bounds={in_bounds}"
        )
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDBufferWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for 64-bit buffer atomics."""

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
        """Executes a buffer_atomic_smin_x2 instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        # 1. Latch inputs for each lane
        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            smin_val = self.read_lane_reg_u64(lane, inst.vdata.index)
            vaddr = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((smin_val, vaddr))

        # 2. Serialize atomic operations across active lanes in ascending order
        lane_results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, False))
                continue

            smin_val, vaddr = lane_inputs[lane]
            ret_val, in_bounds = ref_buffer_atomic_smin_x2(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr,
                smin_val=smin_val,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if inst.mods.glc:
                self.write_lane_reg_u64(lane, inst.vdata.index, ret_val)

            lane_results.append((ret_val, in_bounds))

        return lane_results


class TestBufferAtomicSminX2Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_smin_x2_in_bounds(self) -> None:
        # Pre-seed address 64 with 100
        self.mem.write_u64(64, 100)
        ret, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            smin_val=50,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 50)
        self.assertEqual(self.mem.read_u64(64), 50)

    def test_smin_x2_glc_returns_pre_op_value(self) -> None:
        # Pre-seed address 128 with -10 (0xFFFFFFFFFFFFFFF6)
        val_neg10 = (-10) & MASK64
        val_neg20 = (-20) & MASK64
        self.mem.write_u64(128, val_neg10)
        ret, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            smin_val=val_neg20,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg10)
        self.assertEqual(self.mem.read_u64(128), val_neg20)

    def test_smin_x2_no_update_when_greater(self) -> None:
        # Current is -50, operand is -10; min(-50, -10) = -50
        val_neg50 = (-50) & MASK64
        val_neg10 = (-10) & MASK64
        self.mem.write_u64(32, val_neg50)
        ret, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=32,
            smin_val=val_neg10,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg50)
        self.assertEqual(self.mem.read_u64(32), val_neg50)

    def test_smin_x2_oob_rejection(self) -> None:
        # num_records is 1024. Offset 1020 has 1020 + 8 = 1028 > 1024 (OOB)
        self.mem.write_u64(1020, 0x777788889999AAAA)
        ret, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1020,
            smin_val=0x1111222233334444,
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(1020), 0x777788889999AAAA)

    def test_smin_x2_boundary_exact_fit(self) -> None:
        # Offset 1016 has 1016 + 8 == 1024 (exact fit, in-bounds)
        self.mem.write_u64(1016, 1000)
        ret, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1016,
            smin_val=500,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 1000)
        self.assertEqual(self.mem.read_u64(1016), 500)

    def test_smin_x2_offset_combination(self) -> None:
        # Total offset = soffset(128) + inst_offset(64) + vaddr(8) = 200
        self.mem.write_u64(200, 100)
        ret, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            self.desc,
            soffset=128,
            vaddr=8,
            smin_val=25,
            inst_offset=64,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 100)
        self.assertEqual(self.mem.read_u64(200), 25)

    def test_smin_x2_signed_extremes(self) -> None:
        int64_min = 0x8000000000000000  # -9223372036854775808
        int64_max = 0x7FFFFFFFFFFFFFFF  #  9223372036854775807
        self.mem.write_u64(0, int64_max)
        ret, in_bounds = ref_buffer_atomic_smin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=0,
            smin_val=int64_min,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, int64_max)
        self.assertEqual(self.mem.read_u64(0), int64_min)

    def test_descriptor_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x00007FFF12340000,
            stride=32,
            num_records=131072,
            format_flags=0x40,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc, restored)


class TestBufferAtomicSminX2AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=1)
        v_pair = Register(RegKind.VGPR, 2, count=2)
        s_desc = Register(RegKind.SGPR, 8, count=4)

        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v_pair.format_asm(), "v[2:3]")
        self.assertEqual(s_desc.format_asm(), "s[8:11]")

    def test_instruction_assembly_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=32, glc=True, slc=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_smin_x2 v[0:1], v2, s[4:7], s8 offen offset:32 glc slc",
        )

    def test_immediate_soffset_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 4, count=2),
            vaddr=Register(RegKind.VGPR, 6, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_smin_x2 v[4:5], v6, s[12:15], 0 offen",
        )

    def test_vdata_must_be_vgpr_count_2(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.SGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_srsrc_must_be_sgpr_count_4(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.VGPR, 4, count=4),
                soffset=0,
            )

    def test_soffset_register_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=Register(RegKind.VGPR, 8, count=1),
            )

    def test_immediate_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=4096)
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=-1)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=True),
            )
        )
        expected = "buffer_atomic_smin_x2 v[0:1], v2, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


class TestBufferAtomicSminX2ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_smin_x2_execution(self) -> None:
        self.interp.mem.write_u64(64, 1000)
        self.interp.write_reg_u64(0, 500)             # v[0:1]
        self.interp.write_reg_u32(RegKind.VGPR, 2, 64) # v2: vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(self.interp.mem.read_u64(64), 500)
        # Without glc, v[0:1] untouched
        self.assertEqual(self.interp.read_reg_u64(0), 500)

    def test_scalar_smin_x2_glc_writeback(self) -> None:
        val_neg100 = (-100) & MASK64
        val_neg200 = (-200) & MASK64
        self.interp.mem.write_u64(128, val_neg100)
        self.interp.write_reg_u64(0, val_neg200)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg100)
        # v[0:1] updated to previous memory value (-100)
        self.assertEqual(self.interp.read_reg_u64(0), val_neg100)
        # Memory holds minimum (-200)
        self.assertEqual(self.interp.mem.read_u64(128), val_neg200)

    def test_scalar_smin_x2_no_update_when_greater(self) -> None:
        val_neg500 = (-500) & MASK64
        val_neg100 = (-100) & MASK64
        self.interp.mem.write_u64(32, val_neg500)
        self.interp.write_reg_u64(0, val_neg100)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 32)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg500)
        self.assertEqual(self.interp.mem.read_u64(32), val_neg500)

    def test_scalar_oob_suppression(self) -> None:
        # Offset 1020 is out of bounds for 8-byte access
        self.interp.mem.write_u64(1020, 0x9999888877776666)
        self.interp.write_reg_u64(0, 0x123456789ABCDEF0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 1020)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
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
        self.assertEqual(self.interp.mem.read_u64(1020), 0x9999888877776666)

    def test_scalar_program_chain(self) -> None:
        self.interp.mem.write_u64(0, 1000)
        self.interp.write_reg_u64(0, 500)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)
        self.interp.write_reg_u64(4, 200)

        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=False),
            )
        )
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
                vdata=Register(RegKind.VGPR, 4, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=False),
            )
        )

        results = self.interp.run_program(prog)
        self.assertTrue(results[0][1])
        self.assertTrue(results[1][1])
        self.assertEqual(self.interp.mem.read_u64(0), 200)
        self.assertEqual(len(self.interp.trace), 2)


class TestBufferAtomicSminX2SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=65536,
        )

    def test_wave32_full_active_independent_locations(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        for lane in range(32):
            addr = lane * 8
            simd.mem.write_u64(addr, 1000 + lane)
            simd.write_lane_reg_u64(lane, 0, 500 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(32):
            ret, in_bounds = results[lane]
            self.assertTrue(in_bounds)
            self.assertEqual(ret, 1000 + lane)
            self.assertEqual(simd.mem.read_u64(lane * 8), 500 + lane)
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 1000 + lane)

    def test_wave64_full_active_independent_locations(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        for lane in range(64):
            addr = lane * 8
            val_init = (-10) & MASK64
            val_comp = (-20 - lane) & MASK64
            simd.mem.write_u64(addr, val_init)
            simd.write_lane_reg_u64(lane, 0, val_comp)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(64):
            ret, in_bounds = results[lane]
            self.assertTrue(in_bounds)
            self.assertEqual(ret, (-10) & MASK64)
            self.assertEqual(simd.mem.read_u64(lane * 8), (-20 - lane) & MASK64)

    def test_wavefront_colliding_address_serialization(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x0000000F)  # Lanes 0..3 active

        target_addr = 64
        simd.mem.write_u64(target_addr, 100)

        # Lane 0: 50 -> min(100, 50) = 50
        simd.write_lane_reg_u64(0, 0, 50)
        simd.write_lane_reg_u32(0, RegKind.VGPR, 2, target_addr)

        # Lane 1: 20 -> min(50, 20) = 20
        simd.write_lane_reg_u64(1, 0, 20)
        simd.write_lane_reg_u32(1, RegKind.VGPR, 2, target_addr)

        # Lane 2: 80 -> min(20, 80) = 20 (not smaller)
        simd.write_lane_reg_u64(2, 0, 80)
        simd.write_lane_reg_u32(2, RegKind.VGPR, 2, target_addr)

        # Lane 3: -10 -> min(20, -10) = -10
        val_neg10 = (-10) & MASK64
        simd.write_lane_reg_u64(3, 0, val_neg10)
        simd.write_lane_reg_u32(3, RegKind.VGPR, 2, target_addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        self.assertEqual(results[0][0], 100)
        self.assertEqual(results[1][0], 50)
        self.assertEqual(results[2][0], 20)
        self.assertEqual(results[3][0], 20)
        # Final memory holds -10
        self.assertEqual(simd.mem.read_u64(target_addr), val_neg10)

    def test_partial_exec_masks_checkerboard(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x55555555)  # Even lanes active

        for lane in range(32):
            simd.mem.write_u64(lane * 8, 100)
            simd.write_lane_reg_u64(lane, 0, 50)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane * 8)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        simd.step(inst)

        for lane in range(32):
            expected = 50 if (lane % 2 == 0) else 100
            self.assertEqual(simd.mem.read_u64(lane * 8), expected)

    def test_exec_mask_zero_quiescent(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0)

        simd.mem.write_u64(0, 0x12345678)
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        self.assertEqual(len(results), 64)
        for r in results:
            self.assertIsNone(r[0])
            self.assertFalse(r[1])
        self.assertEqual(simd.mem.read_u64(0), 0x12345678)


class TestBufferAtomicSminX2Aliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_aliasing_vdata0_equals_vaddr_with_glc(self) -> None:
        addr = 64
        self.interp.mem.write_u64(addr, 100)
        # v0 serves as vaddr and low 32 of smin_val (value = 64)
        self.interp.write_reg_u32(RegKind.VGPR, 0, addr)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 100)
        # Memory at 64 was min(100, 64) = 64
        self.assertEqual(self.interp.mem.read_u64(addr), 64)
        # v[0:1] receives pre-op value (100)
        self.assertEqual(self.interp.read_reg_u64(0), 100)

    def test_aliasing_vdata1_equals_vaddr_with_glc(self) -> None:
        addr = 128
        val_huge = 0x7FFFFFFFFFFFFFFF
        self.interp.mem.write_u64(addr, val_huge)
        # v1 serves as vaddr and high 32 of smin_val
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x10)
        self.interp.write_reg_u32(RegKind.VGPR, 1, addr)

        smin_val = (addr << 32) | 0x10

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_huge)
        self.assertEqual(self.interp.mem.read_u64(addr), smin_val)
        self.assertEqual(self.interp.read_reg_u64(0), val_huge)

    def test_unrelated_vgprs_preserved(self) -> None:
        for idx in range(10, 20):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0x77000000 + idx)

        self.interp.write_reg_u64(0, 0x1234)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        for idx in range(10, 20):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), 0x77000000 + idx)

    def test_sgpr_descriptor_and_offset_preserved(self) -> None:
        d0, d1, d2, d3 = self.desc.to_dwords()
        self.interp.write_reg_u32(RegKind.SGPR, 8, 32)  # soffset

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 4), d0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 5), d1)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 6), d2)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 7), d3)
        self.assertEqual(self.interp.read_reg_u32(RegKind.SGPR, 8), 32)

    def test_simd_aliasing_with_divergent_exec_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x0000000F  # Lanes 0..3 active
        simd.set_exec_mask(exec_mask)

        for lane in range(32):
            addr = lane * 8
            simd.mem.write_u64(addr, 1000)
            # Aliasing: vaddr = v0 = addr
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, addr)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Active lanes: memory min(1000, addr) = addr, v[0:1] receives 1000
        for lane in range(4):
            addr = lane * 8
            self.assertEqual(simd.mem.read_u64(addr), addr)
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 1000)

        # Inactive lanes: memory untouched (1000), v[0:1] untouched (addr)
        for lane in range(4, 32):
            addr = lane * 8
            self.assertEqual(simd.mem.read_u64(addr), 1000)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), addr)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), 0)


class TestBufferAtomicSminX2ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_condition_codes_zero_pos_neg(self) -> None:
        # Pre-seed memory at 0 with 0
        self.interp.mem.write_u64(0, 0)
        self.interp.write_reg_u64(0, (-10) & MASK64)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        # Pre-op return value is 0
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)

        # Next operation: memory holds -10 (negative)
        self.interp.write_reg_u64(0, (-50) & MASK64)
        self.interp.step(inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)

        # Reset memory to positive 100
        self.interp.mem.write_u64(0, 100)
        self.interp.write_reg_u64(0, 50)
        self.interp.step(inst)
        cc_pos = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)

    def test_updated_flag_detection(self) -> None:
        self.interp.mem.write_u64(0, 100)
        self.interp.write_reg_u64(0, 50)  # strictly smaller -> updated
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.updated)

        # Greater value -> not updated
        self.interp.write_reg_u64(0, 200)
        self.interp.step(inst)
        cc_same = self.interp.last_condition_codes
        self.assertIsNotNone(cc_same)
        assert cc_same is not None
        self.assertFalse(cc_same.updated)

    def test_status_flags_in_bounds_and_oob(self) -> None:
        self.interp.write_reg_u64(0, 1)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 100)  # in bounds (< 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        flags_in = self.interp.last_status_flags
        self.assertIsNotNone(flags_in)
        assert flags_in is not None
        self.assertTrue(flags_in.in_bounds)
        self.assertFalse(flags_in.oob_fault)

        # OOB access
        self.interp.write_reg_u32(RegKind.VGPR, 2, 2000)
        self.interp.step(inst)
        flags_oob = self.interp.last_status_flags
        self.assertIsNotNone(flags_oob)
        assert flags_oob is not None
        self.assertFalse(flags_oob.in_bounds)
        self.assertTrue(flags_oob.oob_fault)

    def test_carry_borrow_vcc_isolation(self) -> None:
        canary = 0x1234567887654321
        self.interp.vcc = canary

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, canary)


class TestBufferAtomicSminX2BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 bit-pattern preservation, signed extremes, and buffer boundary limits."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_signed_64bit_min_max_extremes(self) -> None:
        int64_min = 0x8000000000000000
        int64_max = 0x7FFFFFFFFFFFFFFF
        self.interp.mem.write_u64(0, int64_max)
        self.interp.write_reg_u64(0, int64_min)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        # min(int64_max, int64_min) -> int64_min
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, int64_max)
        self.assertEqual(self.interp.mem.read_u64(0), int64_min)
        self.assertEqual(self.interp.read_reg_u64(0), int64_max)

    def test_signed_negative_vs_zero_and_minus_one(self) -> None:
        minus_one = 0xFFFFFFFFFFFFFFFF
        zero = 0x0
        self.interp.mem.write_u64(0, zero)
        self.interp.write_reg_u64(0, minus_one)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        # min(0, -1) -> -1
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, zero)
        self.assertEqual(self.interp.mem.read_u64(0), minus_one)

    def test_ieee754_double_sign_bit_and_zero_ordering(self) -> None:
        # IEEE 754 +0.0 is 0x0000000000000000
        # IEEE 754 -0.0 is 0x8000000000000000 (signed int64: -9223372036854775808)
        # In signed integer arithmetic, -0.0 bit pattern < +0.0 bit pattern
        pos_zero = 0x0000000000000000
        neg_zero = 0x8000000000000000
        self.interp.mem.write_u64(0, pos_zero)
        self.interp.write_reg_u64(0, neg_zero)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, pos_zero)
        self.assertEqual(self.interp.mem.read_u64(0), neg_zero)

    def test_ieee754_subnormals_bitwise_integrity_no_flush(self) -> None:
        # Verify subnormal double bit patterns are preserved exactly without flush-to-zero (FTZ)
        pos_subnormal = 0x0000000000000005
        neg_subnormal = 0x8000000000000005
        self.interp.mem.write_u64(0, pos_subnormal)
        self.interp.write_reg_u64(0, neg_subnormal)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, pos_subnormal)
        # Exactly neg_subnormal bitwise, no FTZ
        self.assertEqual(self.interp.mem.read_u64(0), neg_subnormal)

    def test_ieee754_nan_and_inf_no_canonicalization(self) -> None:
        # Double quiet NaN: 0x7FF80000CAFEBABE (positive sign bit -> positive s64)
        # Double -Infinity: 0xFFF0000000000000 (negative sign bit -> negative s64)
        # s64(-Inf) < s64(NaN)
        qnan = 0x7FF80000CAFEBABE
        neg_inf = 0xFFF0000000000000
        self.interp.mem.write_u64(0, qnan)
        self.interp.write_reg_u64(0, neg_inf)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, qnan)
        # Exact payload preserved with no NaN quiet-bit modifications
        self.assertEqual(self.interp.mem.read_u64(0), neg_inf)

    def test_buffer_boundary_exact_fit_and_overrun(self) -> None:
        # Buffer has 1024 records (bytes). 64-bit access is 8 bytes.
        # Offset 1016: 1016 + 8 == 1024 (valid)
        self.interp.mem.write_u64(1016, 500)
        self.interp.write_reg_u64(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 1016)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 500)
        self.assertEqual(self.interp.mem.read_u64(1016), 100)
        self.assertIsNotNone(self.interp.last_status_flags)
        assert self.interp.last_status_flags is not None
        self.assertTrue(self.interp.last_status_flags.in_bounds)

        # Offset 1017: 1017 + 8 == 1025 > 1024 (OOB fault)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 1017)
        ret_oob, in_bounds_oob = self.interp.step(inst)
        self.assertFalse(in_bounds_oob)
        self.assertEqual(ret_oob, 0)
        self.assertIsNotNone(self.interp.last_status_flags)
        assert self.interp.last_status_flags is not None
        self.assertTrue(self.interp.last_status_flags.oob_fault)

    def test_zero_capacity_buffer_rejection(self) -> None:
        zero_desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=0)
        self.interp.set_buffer_descriptor(8, zero_desc)
        self.interp.write_reg_u64(0, 42)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 8, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertIsNotNone(self.interp.last_status_flags)
        assert self.interp.last_status_flags is not None
        self.assertTrue(self.interp.last_status_flags.oob_fault)


if __name__ == "__main__":
    unittest.main()
