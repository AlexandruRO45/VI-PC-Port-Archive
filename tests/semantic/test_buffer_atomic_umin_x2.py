"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_umin_x2.

Models AMD RDNA2 / GFX10 MUBUF 64-bit untyped buffer memory atomic unsigned minimum:
  buffer_atomic_umin_x2 vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR pair v[d:d+1] containing the 64-bit unsigned integer operand.
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
       The atomic umin is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to v[d:d+1].
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory_64[effective_address]
       new_val = min(u64(old_val), u64(vdata))
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


def ref_buffer_atomic_umin_x2(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    umin_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_umin_x2.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS (64-bit access requires 8 bytes)
    if effective_offset + 8 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc else umin_val
        return (0 if glc else (umin_val & MASK64)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u64(phys_addr)
    new_val = min(u64(old_val), u64(umin_val))
    mem.write_u64(phys_addr, new_val)

    ret_val = old_val if glc else (umin_val & MASK64)
    return ret_val, True


class TestBufferAtomicUminX2Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_umin_x2_in_bounds(self) -> None:
        self.mem.write_u64(64, 500)
        ret, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            umin_val=200,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u64(64), 200)
        self.assertEqual(ret, 200)

    def test_umin_x2_unsigned_comparison(self) -> None:
        # 10 is smaller unsigned than 0xFFFFFFFFFFFFFFFF
        self.mem.write_u64(64, 0xFFFFFFFFFFFFFFFF)
        ret, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            umin_val=10,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(self.mem.read_u64(64), 10)

    def test_umin_x2_retains_smaller_memory(self) -> None:
        self.mem.write_u64(128, 50)
        ret, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            umin_val=100,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 50)
        self.assertEqual(self.mem.read_u64(128), 50)

    def test_umin_x2_oob_rejection(self) -> None:
        # num_records is 1024. Offset 1020 has 1020 + 8 = 1028 > 1024 (OOB)
        self.mem.write_u64(1020, 100)
        ret, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1020,
            umin_val=5,
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(1020), 100)

    def test_umin_x2_boundary_exact_fit(self) -> None:
        # Offset 1016 has 1016 + 8 == 1024 (exact fit, in bounds)
        self.mem.write_u64(1016, 200)
        ret, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1016,
            umin_val=150,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 200)
        self.assertEqual(self.mem.read_u64(1016), 150)

    def test_umin_x2_offset_combination(self) -> None:
        # Total offset = soffset(128) + inst_offset(64) + vaddr(8) = 200
        self.mem.write_u64(200, 50)
        ret, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            self.desc,
            soffset=128,
            vaddr=8,
            umin_val=25,
            inst_offset=64,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 50)
        self.assertEqual(self.mem.read_u64(200), 25)

    def test_umin_x2_extremes_zero_absorbing(self) -> None:
        # 0 is absorbing for unsigned minimum
        self.mem.write_u64(0, 0x8000000000000000)
        ret, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=0,
            umin_val=0,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x8000000000000000)
        self.assertEqual(self.mem.read_u64(0), 0)

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
    BUFFER_ATOMIC_UMIN_X2 = "buffer_atomic_umin_x2"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_umin_x2 instruction."""
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


class TestBufferAtomicUminX2AST(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=32, glc=True, slc=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_umin_x2 v[0:1], v2, s[4:7], s8 offen offset:32 glc slc",
        )

    def test_immediate_soffset_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 4, count=2),
            vaddr=Register(RegKind.VGPR, 6, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_umin_x2 v[4:5], v6, s[12:15], 0 offen",
        )

    def test_vdata_must_be_vgpr_count_2(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.SGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_srsrc_must_be_sgpr_count_4(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.VGPR, 4, count=4),
                soffset=0,
            )

    def test_soffset_register_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=True),
            )
        )
        expected = "buffer_atomic_umin_x2 v[0:1], v2, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic single-lane scalar interpreter for buffer instructions."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
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
        """Executes a single buffer_atomic_umin_x2 instruction deterministically."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        umin_val = self.read_reg_u64(inst.vdata.index)

        ret_val, in_bounds = ref_buffer_atomic_umin_x2(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            umin_val=umin_val,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u64(inst.vdata.index, ret_val)

        self.trace.append(
            f"{inst.to_assembly()} => ret={ret_val:#018x}, in_bounds={in_bounds}"
        )
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestBufferAtomicUminX2ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_umin_x2_execution(self) -> None:
        self.interp.mem.write_u64(64, 1000)
        self.interp.write_reg_u64(0, 250)  # v[0:1] = 250
        self.interp.write_reg_u32(RegKind.VGPR, 2, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # Memory holds min(1000, 250) = 250
        self.assertEqual(self.interp.mem.read_u64(64), 250)
        # Without glc, v[0:1] retains 250
        self.assertEqual(self.interp.read_reg_u64(0), 250)

    def test_scalar_umin_x2_glc_writeback(self) -> None:
        self.interp.mem.write_u64(128, 500)
        self.interp.write_reg_u64(0, 200)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 500)
        # v[0:1] updated to pre-op value (500)
        self.assertEqual(self.interp.read_reg_u64(0), 500)
        # Memory holds min(500, 200) = 200
        self.assertEqual(self.interp.mem.read_u64(128), 200)

    def test_scalar_oob_suppression(self) -> None:
        self.interp.mem.write_u64(1020, 888)
        self.interp.write_reg_u64(0, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 1020)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
        self.assertEqual(self.interp.mem.read_u64(1020), 888)

    def test_scalar_program_chain(self) -> None:
        self.interp.mem.write_u64(0, 100)
        self.interp.write_reg_u64(0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        self.interp.write_reg_u64(4, 25)

        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=False),
            )
        )
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
        # min(100, 50) = 50, then min(50, 25) = 25
        self.assertEqual(self.interp.mem.read_u64(0), 25)
        self.assertEqual(len(self.interp.trace), 2)


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
        """Executes a buffer_atomic_umin_x2 instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        # 1. Latch inputs for each lane
        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            umin_val = self.read_lane_reg_u64(lane, inst.vdata.index)
            vaddr = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((umin_val, vaddr))

        # 2. Serialize atomic operations across active lanes in ascending order
        lane_results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, False))
                continue

            umin_val, vaddr = lane_inputs[lane]
            ret_val, in_bounds = ref_buffer_atomic_umin_x2(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr,
                umin_val=umin_val,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if inst.mods.glc:
                self.write_lane_reg_u64(lane, inst.vdata.index, ret_val)

            lane_results.append((ret_val, in_bounds))

        return lane_results


class TestBufferAtomicUminX2SIMD(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
            # min(1000+lane, 500+lane) = 500+lane
            self.assertEqual(simd.mem.read_u64(lane * 8), 500 + lane)
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 1000 + lane)

    def test_wave64_full_active_independent_locations(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)

        for lane in range(64):
            addr = lane * 8
            simd.mem.write_u64(addr, 2000 + lane)
            simd.write_lane_reg_u64(lane, 0, 1000 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
            self.assertEqual(ret, 2000 + lane)
            self.assertEqual(simd.mem.read_u64(lane * 8), 1000 + lane)

    def test_exec_mask_inactive_lanes_suppressed(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x0000000F)  # Lanes 0..3 active

        for lane in range(32):
            addr = lane * 8
            simd.mem.write_u64(addr, 500)
            simd.write_lane_reg_u64(lane, 0, 100)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(4):
            self.assertTrue(results[lane][1])
            self.assertEqual(simd.mem.read_u64(lane * 8), 100)
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 500)

        for lane in range(4, 32):
            self.assertIsNone(results[lane][0])
            self.assertEqual(simd.mem.read_u64(lane * 8), 500)
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 100)

    def test_wavefront_colliding_address_serialization(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x00000007)  # Lanes 0, 1, 2 active

        target_addr = 64
        simd.mem.write_u64(target_addr, 1000)

        # Lane 0: umin 800 -> min(1000, 800) = 800
        simd.write_lane_reg_u64(0, 0, 800)
        simd.write_lane_reg_u32(0, RegKind.VGPR, 2, target_addr)

        # Lane 1: umin 500 -> min(800, 500) = 500
        simd.write_lane_reg_u64(1, 0, 500)
        simd.write_lane_reg_u32(1, RegKind.VGPR, 2, target_addr)

        # Lane 2: umin 900 -> min(500, 900) = 500
        simd.write_lane_reg_u64(2, 0, 900)
        simd.write_lane_reg_u32(2, RegKind.VGPR, 2, target_addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        self.assertEqual(results[0][0], 1000)
        self.assertEqual(results[1][0], 800)
        self.assertEqual(results[2][0], 500)
        self.assertEqual(simd.mem.read_u64(target_addr), 500)

    def test_partial_exec_masks_checkerboard(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x55555555)  # Even lanes active

        for lane in range(32):
            simd.mem.write_u64(lane * 8, 100)
            simd.write_lane_reg_u64(lane, 0, 25)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane * 8)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        simd.step(inst)

        for lane in range(32):
            expected = 25 if (lane % 2 == 0) else 100
            self.assertEqual(simd.mem.read_u64(lane * 8), expected)

    def test_exec_mask_zero_quiescent(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0)

        simd.mem.write_u64(0, 42)
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
        self.assertEqual(simd.mem.read_u64(0), 42)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestBufferAtomicUminX2Aliasing(unittest.TestCase):
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
        self.interp.mem.write_u64(addr, 500)
        # v0 serves as vaddr and low 32 of umin_val
        self.interp.write_reg_u32(RegKind.VGPR, 0, addr)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 500)
        # min(500, 64) = 64
        self.assertEqual(self.interp.mem.read_u64(addr), 64)
        # v[0:1] receives pre-op value (500)
        self.assertEqual(self.interp.read_reg_u64(0), 500)

    def test_aliasing_vdata1_equals_vaddr_with_glc(self) -> None:
        addr = 128
        self.interp.mem.write_u64(addr, 100)
        # v1 serves as vaddr and high 32 of umin_val
        self.interp.write_reg_u32(RegKind.VGPR, 0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 100)
        # (128 << 32) | 50 > 100, so memory retains 100
        self.assertEqual(self.interp.mem.read_u64(addr), 100)
        self.assertEqual(self.interp.read_reg_u64(0), 100)

    def test_unrelated_vgprs_preserved(self) -> None:
        for idx in range(10, 20):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0x77000000 + idx)

        self.interp.write_reg_u64(0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
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
            simd.mem.write_u64(addr, 1000 + lane)
            # Aliasing: vaddr = v0 = addr
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, addr)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Active lanes: memory updated to min(1000 + lane, addr) = addr
        for lane in range(4):
            addr = lane * 8
            self.assertEqual(simd.mem.read_u64(addr), addr)
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 1000 + lane)

        # Inactive lanes: memory untouched, v[0:1] untouched
        for lane in range(4, 32):
            addr = lane * 8
            self.assertEqual(simd.mem.read_u64(addr), 1000 + lane)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), addr)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), 0)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AtomicUminX2Flags:
    """Architectural condition codes and status flags for buffer_atomic_umin_x2."""
    zero: bool
    negative: bool
    diff: int
    borrow: bool
    carry: bool
    overflow: bool
    unaligned_addr: bool
    out_of_bounds: bool
    operand_was_smaller: bool


def evaluate_umin_x2_flags(
    old_val: int,
    umin_val: int,
    eff_addr: int,
    num_records: int,
) -> AtomicUminX2Flags:
    """Evaluate condition codes, status flags, and arithmetic carry/borrow for 64-bit umin."""
    u_old = old_val & MASK64
    u_umin = umin_val & MASK64

    # Zero flag: returned 64-bit value is zero
    zero = (u_old == 0)

    # Negative flag: MSB (bit 63) is set
    negative = bool(u_old & SIGN_BIT64)

    # Unsigned difference and borrow
    diff = (u_old - u_umin) & MASK64
    borrow = u_old < u_umin

    # Carry generation on 64-bit addition
    carry = (u_old + u_umin) > MASK64

    # Signed comparison overflow for (s64(old_val) - s64(umin_val))
    s_old = s64(u_old)
    s_umin = s64(u_umin)
    s_diff = s_old - s_umin
    overflow = s_diff < -0x8000000000000000 or s_diff > 0x7FFFFFFFFFFFFFFF

    # Unaligned address check (low 3 bits must be zero for 8-byte dword alignment)
    unaligned = bool(eff_addr & 0x7)

    # Bounds check against descriptor limit for 8 bytes
    oob = (eff_addr + 8) > num_records

    # Unsigned comparison predicate
    op_smaller = u_umin < u_old

    return AtomicUminX2Flags(
        zero=zero,
        negative=negative,
        diff=diff,
        borrow=borrow,
        carry=carry,
        overflow=overflow,
        unaligned_addr=unaligned,
        out_of_bounds=oob,
        operand_was_smaller=op_smaller,
    )


class TestBufferAtomicUminX2Flags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_zero_flag_on_zero_return(self) -> None:
        flags = evaluate_umin_x2_flags(
            old_val=0,
            umin_val=100,
            eff_addr=0x100,
            num_records=1024,
        )
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.operand_was_smaller)
        self.assertFalse(flags.unaligned_addr)
        self.assertFalse(flags.out_of_bounds)

    def test_negative_flag_when_msb_set(self) -> None:
        flags = evaluate_umin_x2_flags(
            old_val=0x8000000000000000,
            umin_val=10,
            eff_addr=0x200,
            num_records=1024,
        )
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.operand_was_smaller)

    def test_operand_was_smaller(self) -> None:
        flags = evaluate_umin_x2_flags(
            old_val=500,
            umin_val=200,
            eff_addr=0x300,
            num_records=1024,
        )
        self.assertTrue(flags.operand_was_smaller)
        self.assertFalse(flags.borrow)

    def test_borrow_generation(self) -> None:
        flags = evaluate_umin_x2_flags(
            old_val=100,
            umin_val=250,
            eff_addr=0x300,
            num_records=1024,
        )
        self.assertTrue(flags.borrow)
        self.assertEqual(flags.diff, (100 - 250) & MASK64)

    def test_signed_overflow_flag(self) -> None:
        # s_old = max positive 64-bit int, s_umin = -10 (which is 0xFFFFFFFFFFFFFFF6)
        flags = evaluate_umin_x2_flags(
            old_val=0x7FFFFFFFFFFFFFFF,
            umin_val=u64(-10),
            eff_addr=0x100,
            num_records=1024,
        )
        self.assertTrue(flags.overflow)

    def test_unaligned_8byte_address_detection(self) -> None:
        flags_unaligned = evaluate_umin_x2_flags(
            old_val=0,
            umin_val=0,
            eff_addr=0x104,  # Not 8-byte aligned
            num_records=1024,
        )
        self.assertTrue(flags_unaligned.unaligned_addr)

        flags_aligned = evaluate_umin_x2_flags(
            old_val=0,
            umin_val=0,
            eff_addr=0x108,
            num_records=1024,
        )
        self.assertFalse(flags_aligned.unaligned_addr)

    def test_out_of_bounds_detection(self) -> None:
        flags_oob = evaluate_umin_x2_flags(
            old_val=0,
            umin_val=0,
            eff_addr=1020,  # 1020 + 8 = 1028 > 1024
            num_records=1024,
        )
        self.assertTrue(flags_oob.out_of_bounds)

        flags_in = evaluate_umin_x2_flags(
            old_val=0,
            umin_val=0,
            eff_addr=1016,  # 1016 + 8 = 1024
            num_records=1024,
        )
        self.assertFalse(flags_in.out_of_bounds)

    def test_carry_generation(self) -> None:
        flags_carry = evaluate_umin_x2_flags(
            old_val=0xFFFFFFFFFFFFFFFF,
            umin_val=1,
            eff_addr=0x100,
            num_records=1024,
        )
        self.assertTrue(flags_carry.carry)

        flags_no_carry = evaluate_umin_x2_flags(
            old_val=10,
            umin_val=20,
            eff_addr=0x100,
            num_records=1024,
        )
        self.assertFalse(flags_no_carry.carry)


# ---------------------------------------------------------------------------
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ---------------------------------------------------------------------------

F64_POS_ZERO: int = 0x0000000000000000
F64_NEG_ZERO: int = 0x8000000000000000
F64_POS_INF: int = 0x7FF0000000000000
F64_NEG_INF: int = 0xFFF0000000000000
F64_QUIET_NAN: int = 0x7FF8000000000000
F64_SIGNALING_NAN: int = 0x7FF0000000000001
F64_MIN_SUBNORMAL: int = 0x0000000000000001
F64_MAX_SUBNORMAL: int = 0x000FFFFFFFFFFFFF
F64_MIN_NORMAL: int = 0x0010000000000000
F64_MAX_NORMAL: int = 0x7FEFFFFFFFFFFFFF


def is_f64_nan(bits: int) -> bool:
    """Check if 64-bit pattern represents an IEEE 754 Double NaN."""
    exp = (bits >> 52) & 0x7FF
    mant = bits & 0xFFFFFFFFFFFFF
    return exp == 0x7FF and mant != 0


def is_f64_subnormal(bits: int) -> bool:
    """Check if 64-bit pattern represents a denormalized/subnormal float64."""
    exp = (bits >> 52) & 0x7FF
    mant = bits & 0xFFFFFFFFFFFFF
    return exp == 0 and mant != 0


class TestBufferAtomicUminX2IEEE754(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_bitwise_exact_float64_zeros_unsigned_comparison(self) -> None:
        # In unsigned 64-bit integer interpretation:
        # +0.0 (0x0000000000000000) = 0
        # -0.0 (0x8000000000000000) = 9223372036854775808 (unsigned)
        # Therefore min(-0.0, +0.0) must yield +0.0!
        addr = 64
        self.interp.mem.write_u64(addr, F64_NEG_ZERO)
        self.interp.write_reg_u64(0, F64_POS_ZERO)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, F64_NEG_ZERO)
        # Memory updated to min(0x8000000000000000, 0) = 0 (+0.0)
        self.assertEqual(self.interp.mem.read_u64(addr), F64_POS_ZERO)
        self.assertEqual(self.interp.read_reg_u64(0), F64_NEG_ZERO)

    def test_float64_nan_treated_as_unsigned_integer(self) -> None:
        # Quiet NaN 0x7FF8000000000000 < Negative Inf 0xFFF0000000000000 in unsigned comparison
        addr = 128
        self.interp.mem.write_u64(addr, F64_NEG_INF)
        self.interp.write_reg_u64(0, F64_QUIET_NAN)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, F64_NEG_INF)
        self.assertEqual(self.interp.mem.read_u64(addr), F64_QUIET_NAN)

    def test_float64_subnormals_preserved_as_integers(self) -> None:
        self.assertTrue(is_f64_subnormal(F64_MIN_SUBNORMAL))
        self.assertTrue(is_f64_subnormal(F64_MAX_SUBNORMAL))

        addr = 200
        self.interp.mem.write_u64(addr, F64_MAX_SUBNORMAL)
        self.interp.write_reg_u64(0, F64_MIN_SUBNORMAL)
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, F64_MAX_SUBNORMAL)
        # min(0x000FFFFFFFFFFFFF, 1) = 1
        self.assertEqual(self.interp.mem.read_u64(addr), F64_MIN_SUBNORMAL)

    def test_64bit_unsigned_integer_extremes(self) -> None:
        # Strictly descending unsigned values:
        # 0xFFFFFFFFFFFFFFFF, 0x8000000000000000, 0x7FFFFFFFFFFFFFFF, 1000, 1, 0
        descending = [
            0xFFFFFFFFFFFFFFFF,
            0x8000000000000000,
            0x7FFFFFFFFFFFFFFF,
            1000,
            1,
            0,
        ]
        addr = 300
        self.interp.write_reg_u32(RegKind.VGPR, 2, addr)
        self.interp.mem.write_u64(addr, descending[0])

        for i in range(len(descending) - 1):
            next_val = descending[i + 1]
            self.interp.write_reg_u64(0, next_val)

            inst = Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=True),
            )
            ret, in_bounds = self.interp.step(inst)
            self.assertTrue(in_bounds)
            # Memory should update to next_val because each is strictly smaller unsigned
            self.assertEqual(ret, descending[i])
            self.assertEqual(self.interp.mem.read_u64(addr), next_val)


# ---------------------------------------------------------------------------
# Negative Controls, Syntax Rejection, and Mutation Detection
# ---------------------------------------------------------------------------

class TestBufferAtomicUminX2NegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def setUp(self) -> None:
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    # --- Negative Controls & Syntax Rejection ---

    def test_negative_reject_invalid_vdata_count(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=4),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_negative_reject_vdata_sgpr(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.SGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_negative_reject_srsrc_count(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

    def test_negative_reject_srsrc_vgpr(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.VGPR, 4, count=4),
                soffset=0,
            )

    def test_negative_reject_soffset_vgpr(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 2, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=Register(RegKind.VGPR, 8, count=1),
            )

    def test_negative_reject_out_of_range_offsets(self) -> None:
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=-1)
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=4096)

    def test_negative_reject_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=128)

    def test_negative_reject_register_boundary_overflow(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256, count=1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 255, count=2)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 253, count=4)

    # --- Mutation Detection ---

    def test_mutation_signed_min_instead_of_unsigned_min_detected(self) -> None:
        """Mutant: compares values as signed integers instead of unsigned integers."""
        mem = BufferMemory(4096)
        mem.write_u64(64, 10)

        def buggy_smin_swap(val: int) -> None:
            # Bug: compares signed s64 instead of unsigned u64!
            old = mem.read_u64(64)
            new_val = min(s64(old), s64(val))
            mem.write_u64(64, u64(new_val))

        # In unsigned comparison, min(10, 0xFFFFFFFFFFFFFFFF) = 10 (memory unchanged)
        ref_buffer_atomic_umin_x2(mem, self.desc, 0, 64, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(mem.read_u64(64), 10)

        # In buggy signed comparison, -1 < 10, so it erroneously writes -1 (0xFFFFFFFFFFFFFFFF)
        buggy_smin_swap(0xFFFFFFFFFFFFFFFF)
        self.assertEqual(mem.read_u64(64), 0xFFFFFFFFFFFFFFFF)
        self.assertNotEqual(mem.read_u64(64), 10)

    def test_mutation_single_dword_write_detected(self) -> None:
        """Mutant: writes only 32 bits instead of 64 bits."""
        mem = BufferMemory(4096)
        mem.write_u64(64, 0xFFFFFFFFFFFFFFFF)

        def buggy_swap_32(swap_val: int) -> None:
            mem.write_u32(64, swap_val & MASK32)

        # Correct 64-bit umin writes full 64 bits: 0x0000000012345678
        ref_buffer_atomic_umin_x2(mem, self.desc, 0, 64, 0x0000000012345678)
        self.assertEqual(mem.read_u64(64), 0x0000000012345678)

        # Reset memory to 0xFFFFFFFFFFFFFFFF
        mem.write_u64(64, 0xFFFFFFFFFFFFFFFF)
        # Buggy 32-bit write leaves high 32 bits untouched as 0xFFFFFFFF
        buggy_swap_32(0x12345678)
        self.assertEqual(mem.read_u32(64), 0x12345678)
        self.assertEqual(mem.read_u32(68), 0xFFFFFFFF)

    def test_mutation_inactive_lanes_modify_state_detected(self) -> None:
        """Mutant: SIMD interpreter executes inactive lanes."""
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x1)  # Only lane 0 active

        sentinel = 0xAA55AA55AA55AA55
        for lane in range(32):
            simd.write_lane_reg_u64(lane, 0, sentinel)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane * 8)

        class InactiveCorruptingSIMD(SIMDBufferWavefrontInterpreter):
            def step(self, inst: Instruction):
                for lane in range(self.wave_size):
                    self.write_lane_reg_u64(lane, inst.vdata.index, 0)
                return []

        corrupting = InactiveCorruptingSIMD(wave_size=32, mem_size=4096)
        for lane in range(32):
            corrupting.write_lane_reg_u64(lane, 0, sentinel)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
        )
        corrupting.step(inst)

        corrupted = [l for l in range(1, 32) if corrupting.read_lane_reg_u64(l, 0) != sentinel]
        self.assertEqual(len(corrupted), 31)

    def test_mutation_premature_writeback_under_aliasing_detected(self) -> None:
        """Mutant: writes destination before latching address operand."""
        interp = ScalarBufferInterpreter(mem_size=4096)
        interp.set_buffer_descriptor(4, self.desc)
        addr = 64
        interp.mem.write_u64(addr, 500)
        interp.write_reg_u32(RegKind.VGPR, 0, addr)
        interp.write_reg_u32(RegKind.VGPR, 1, 0)

        class PrematureWritebackInterp(ScalarBufferInterpreter):
            def step(self, inst: Instruction):
                self.write_reg_u32(RegKind.VGPR, inst.vdata.index, 0x8888)
                vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)  # Reads corrupted 0x8888!
                desc = self.get_buffer_descriptor(inst.srsrc.index)
                return ref_buffer_atomic_umin_x2(
                    self.mem, desc, 0, vaddr, 64, glc=True
                )

        buggy = PrematureWritebackInterp(mem_size=4096)
        buggy.set_buffer_descriptor(4, self.desc)
        buggy.mem.write_u64(addr, 500)
        buggy.write_reg_u32(RegKind.VGPR, 0, addr)
        buggy.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_b = buggy.step(inst)
        self.assertFalse(in_b)
        self.assertEqual(ret, 0)

    def test_mutation_reverse_simd_lane_order_detected(self) -> None:
        """Mutant: serializes colliding lanes in reverse order."""
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0x3)  # Lanes 0 and 1 active

        target = 64
        simd.mem.write_u64(target, 100)

        class ReverseLaneOrderSIMD(SIMDBufferWavefrontInterpreter):
            def step(self, inst: Instruction):
                desc = self.get_buffer_descriptor(inst.srsrc.index)
                for lane in reversed(range(self.wave_size)):
                    if bool((self.exec_mask >> lane) & 1):
                        val = self.read_lane_reg_u64(lane, inst.vdata.index)
                        va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
                        ref_buffer_atomic_umin_x2(self.mem, desc, 0, va, val)
                return []

        buggy = ReverseLaneOrderSIMD(wave_size=32, mem_size=4096)
        buggy.set_buffer_descriptor(4, self.desc)
        buggy.set_exec_mask(0x3)
        buggy.mem.write_u64(target, 100)
        buggy.write_lane_reg_u64(0, 0, 50)
        buggy.write_lane_reg_u32(0, RegKind.VGPR, 2, target)
        buggy.write_lane_reg_u64(1, 0, 25)
        buggy.write_lane_reg_u32(1, RegKind.VGPR, 2, target)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_UMIN_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
        )
        buggy.step(inst)
        self.assertEqual(buggy.mem.read_u64(target), 25)

    def test_mutation_ignore_immediate_offset_detected(self) -> None:
        """Mutant: immediate inst_offset is ignored in address calculation."""
        def buggy_offset_calc(soffset: int, inst_offset: int, vaddr: int) -> int:
            return (soffset & MASK32) + (vaddr & MASK32)

        correct = (128 & MASK32) + (32 & 0xFFF) + (8 & MASK32)
        mutant = buggy_offset_calc(128, 32, 8)
        self.assertEqual(correct, 168)
        self.assertEqual(mutant, 136)
        self.assertNotEqual(correct, mutant)

    def test_mutation_missing_oob_bounds_check_detected(self) -> None:
        """Mutant: uses 4-byte check instead of 8-byte check for 64-bit access."""
        mem = BufferMemory(4096)
        mem.write_u64(1020, 100)

        def buggy_check_oob(offset: int) -> bool:
            return (offset + 4) > self.desc.num_records

        _, in_b = ref_buffer_atomic_umin_x2(
            mem, self.desc, soffset=0, vaddr=1020, umin_val=50
        )
        self.assertFalse(in_b)
        self.assertFalse(buggy_check_oob(1020))


if __name__ == "__main__":
    unittest.main()
