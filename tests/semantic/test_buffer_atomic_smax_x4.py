"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_smax_x4.

Models RDNA2 / GFX10 128-bit (4-DWORD) untyped buffer memory atomic signed maximum:
  buffer_atomic_smax_x4 vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR quad v[d:d+3] containing the 128-bit signed integer operand
       (packed in little-endian order: v[d]=b31..0, v[d+1]=b63..32, v[d+2]=b95..64, v[d+3]=b127..96).
       If 'glc' is enabled, the 128-bit pre-operation value at memory is returned to v[d:d+3].
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
     If effective_offset + 16 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic smax is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to v[d:d+3].
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory_128[effective_address]
       new_val = max(s128(old_val), s128(vdata))
       memory_128[effective_address] = u128(new_val)
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
MASK128: int = (1 << 128) - 1
SIGN_BIT128: int = 1 << 127


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def u128(val: int) -> int:
    """Normalize integer to unsigned 128-bit range [0, 2^128 - 1]."""
    return val & MASK128


def s128(val: int) -> int:
    """Interpret a 128-bit unsigned integer as signed two's-complement."""
    val &= MASK128
    return val - (1 << 128) if (val & SIGN_BIT128) else val


def pack_u128(d0: int, d1: int, d2: int, d3: int) -> int:
    """Pack four 32-bit dwords into a 128-bit integer (little-endian)."""
    return ((d3 & MASK32) << 96) | ((d2 & MASK32) << 64) | ((d1 & MASK32) << 32) | (d0 & MASK32)


def unpack_u128(val: int) -> tuple[int, int, int, int]:
    """Unpack a 128-bit integer into four 32-bit dwords (little-endian)."""
    val &= MASK128
    return (
        val & MASK32,
        (val >> 32) & MASK32,
        (val >> 64) & MASK32,
        (val >> 96) & MASK32,
    )


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

    def read_u128(self, addr: int) -> int:
        if addr < 0 or addr + 16 > len(self.data):
            return 0
        lo, hi = struct.unpack_from("<QQ", self.data, addr)
        return (hi << 64) | lo

    def write_u128(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 16 > len(self.data):
            return
        lo = val & MASK64
        hi = (val >> 64) & MASK64
        struct.pack_into("<QQ", self.data, addr, lo, hi)


def ref_buffer_atomic_smax_x4(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    smax_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_smax_x4."""
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check: 128-bit access requires 16 contiguous bytes
    if effective_offset + 16 > desc.num_records:
        return (0 if glc else (smax_val & MASK128)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u128(phys_addr)
    new_val = max(s128(old_val), s128(smax_val))
    mem.write_u128(phys_addr, u128(new_val))

    ret_val = old_val if glc else (smax_val & MASK128)
    return ret_val, True

# ---------------------------------------------------------------------------
# Symbolic AST Definitions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    SGPR = "s"
    VGPR = "v"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand."""
    kind: RegKind
    index: int
    count: int = 1

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count == 1:
            return f"{self.kind.value}{self.index}"
        return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"


@dataclasses.dataclass(frozen=True)
class BufferAtomicModifiers:
    """MUBUF atomic instruction modifiers."""
    offen: bool = True
    idxen: bool = False
    glc: bool = False
    slc: bool = False
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0 or self.offset > 4095:
            raise ValueError(f"Immediate inst_offset must be in range [0, 4095]: {self.offset}")


class Opcode(enum.Enum):
    BUFFER_ATOMIC_SMAX_X4 = "buffer_atomic_smax_x4"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_smax_x4 instruction."""
    opcode: Opcode
    vdata: Register
    vaddr: Register
    srsrc: Register
    soffset: Register | int
    mods: BufferAtomicModifiers = dataclasses.field(default_factory=BufferAtomicModifiers)

    def __post_init__(self) -> None:
        if not self.vdata.is_vgpr or self.vdata.count != 4:
            raise ValueError(f"vdata must be a 128-bit VGPR quad (count=4): {self.vdata}")
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


class TestBufferAtomicSmaxX4AST(unittest.TestCase):
    """Unit tests for symbolic AST nodes and assembly formatting."""

    def test_instruction_creation_defaults(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 8, count=4),
            soffset=0,
        )
        self.assertEqual(inst.opcode, Opcode.BUFFER_ATOMIC_SMAX_X4)
        self.assertEqual(inst.vdata.format_asm(), "v[0:3]")
        self.assertEqual(inst.vaddr.format_asm(), "v4")
        self.assertEqual(inst.srsrc.format_asm(), "s[8:11]")
        self.assertEqual(inst.to_assembly(), "buffer_atomic_smax_x4 v[0:3], v4, s[8:11], 0 offen")

    def test_instruction_with_all_modifiers(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 4, count=4),
            vaddr=Register(RegKind.VGPR, 8, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=Register(RegKind.SGPR, 16, count=1),
            mods=BufferAtomicModifiers(offen=True, idxen=True, glc=True, slc=True, offset=256),
        )
        asm = inst.to_assembly()
        self.assertIn("v[4:7]", asm)
        self.assertIn("v8", asm)
        self.assertIn("s[12:15]", asm)
        self.assertIn("s16", asm)
        self.assertIn("offset:256", asm)
        self.assertIn("glc", asm)
        self.assertIn("slc", asm)

    def test_program_representation(self) -> None:
        prog = Program()
        inst1 = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
        )
        inst2 = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 8, count=4),
            vaddr=Register(RegKind.VGPR, 12, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=64,
            mods=BufferAtomicModifiers(glc=True),
        )
        prog.append(inst1)
        prog.append(inst2)
        lines = prog.to_assembly().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("v[0:3]", lines[0])
        self.assertIn("glc", lines[1])

# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Single-lane scalar interpreter for buffer_atomic_smax_x4."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.vcc: int = 0
        self.trace: list[str] = []
        self.last_condition_codes: BufferAtomicConditionCodes | None = None
        self.last_status_flags: BufferStatusFlags | None = None

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

    def read_reg_u128(self, idx: int) -> int:
        d0 = self.read_reg_u32(RegKind.VGPR, idx)
        d1 = self.read_reg_u32(RegKind.VGPR, idx + 1)
        d2 = self.read_reg_u32(RegKind.VGPR, idx + 2)
        d3 = self.read_reg_u32(RegKind.VGPR, idx + 3)
        return pack_u128(d0, d1, d2, d3)

    def write_reg_u128(self, idx: int, val: int) -> None:
        d0, d1, d2, d3 = unpack_u128(val)
        self.write_reg_u32(RegKind.VGPR, idx, d0)
        self.write_reg_u32(RegKind.VGPR, idx + 1, d1)
        self.write_reg_u32(RegKind.VGPR, idx + 2, d2)
        self.write_reg_u32(RegKind.VGPR, idx + 3, d3)

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
        """Executes a single buffer_atomic_smax_x4 instruction deterministically."""
        smax_val = self.read_reg_u128(inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        eff_off = (soffset & MASK32) + (inst.mods.offset & 0xFFF) + (vaddr if inst.mods.offen else 0)
        in_bounds_flag = (eff_off + 16 <= desc.num_records)
        old_val = self.mem.read_u128((desc.base_address + eff_off) & MASK64) if in_bounds_flag else 0

        ret_val, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            smax_val=smax_val,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u128(inst.vdata.index, ret_val)

        if "compute_buffer_condition_codes" in globals():
            self.last_condition_codes = compute_buffer_condition_codes(
                old_val if inst.mods.glc else ret_val, smax_val, ret_val
            )
            self.last_status_flags = compute_buffer_status_flags(in_bounds)

        self.trace.append(
            f"{inst.to_assembly()} => ret={ret_val:#034x}, in_bounds={in_bounds}"
        )
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        return [self.step(inst) for inst in prog.instructions]


class TestBufferAtomicSmaxX4ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_smax_x4_no_glc(self) -> None:
        self.interp.mem.write_u128(64, 100)
        self.interp.write_reg_u128(0, 300)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 300)
        self.assertEqual(self.interp.mem.read_u128(64), 300)
        self.assertEqual(self.interp.read_reg_u128(0), 300)

    def test_scalar_smax_x4_with_glc(self) -> None:
        self.interp.mem.write_u128(128, 500)
        self.interp.write_reg_u128(0, 800)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 500)
        self.assertEqual(self.interp.mem.read_u128(128), 800)
        self.assertEqual(self.interp.read_reg_u128(0), 500)

    def test_scalar_sgpr_soffset(self) -> None:
        self.interp.mem.write_u128(200, 10)
        self.interp.write_reg_u128(0, 20)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 100)
        self.interp.write_reg_u32(RegKind.SGPR, 8, 100)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 10)
        self.assertEqual(self.interp.mem.read_u128(200), 20)

    def test_scalar_immediate_offset(self) -> None:
        self.interp.mem.write_u128(300, 42)
        self.interp.write_reg_u128(0, 99)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 200)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offset=100, glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 42)
        self.assertEqual(self.interp.mem.read_u128(300), 99)

    def test_scalar_oob_glc_clears_destination(self) -> None:
        self.interp.write_reg_u128(0, pack_u128(1, 2, 3, 4))
        self.interp.write_reg_u32(RegKind.VGPR, 4, 2000)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.interp.read_reg_u128(0), 0)

class TestBufferAtomicSmaxX4Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_smax_x4_positive_values(self) -> None:
        val_100 = 100
        val_250 = 250
        self.mem.write_u128(64, val_100)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            smax_val=val_250,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_250)
        self.assertEqual(self.mem.read_u128(64), val_250)

    def test_basic_smax_x4_negative_vs_positive(self) -> None:
        val_neg50 = (-50) & MASK128
        val_pos20 = 20
        self.mem.write_u128(64, val_neg50)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            smax_val=val_pos20,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg50)
        self.assertEqual(self.mem.read_u128(64), val_pos20)

    def test_basic_smax_x4_two_negatives(self) -> None:
        val_neg100 = (-100) & MASK128
        val_neg20 = (-20) & MASK128
        self.mem.write_u128(128, val_neg100)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            smax_val=val_neg20,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg100)
        self.assertEqual(self.mem.read_u128(128), val_neg20)

    def test_smax_x4_no_update_when_smaller(self) -> None:
        val_500 = 500
        val_200 = 200
        self.mem.write_u128(32, val_500)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=32,
            smax_val=val_200,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_500)
        self.assertEqual(self.mem.read_u128(32), val_500)

    def test_smax_x4_oob_rejection(self) -> None:
        canary = pack_u128(0x11111111, 0x22222222, 0x33333333, 0x44444444)
        self.mem.write_u128(1010, canary)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1010,
            smax_val=pack_u128(0x55555555, 0x66666666, 0x77777777, 0x88888888),
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u128(1010), canary)

    def test_smax_x4_boundary_exact_fit(self) -> None:
        self.mem.write_u128(1008, 1000)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1008,
            smax_val=5000,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 1000)
        self.assertEqual(self.mem.read_u128(1008), 5000)

    def test_smax_x4_offset_combination(self) -> None:
        self.mem.write_u128(208, 1234)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=128,
            vaddr=16,
            smax_val=5678,
            inst_offset=64,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 1234)
        self.assertEqual(self.mem.read_u128(208), 5678)

    def test_pack_unpack_u128_roundtrip(self) -> None:
        d0, d1, d2, d3 = 0xDEADBEEF, 0xCAFEBABE, 0x01234567, 0x89ABCDEF
        packed = pack_u128(d0, d1, d2, d3)
        unpacked = unpack_u128(packed)
        self.assertEqual(unpacked, (d0, d1, d2, d3))

if __name__ == "__main__":
    unittest.main()
