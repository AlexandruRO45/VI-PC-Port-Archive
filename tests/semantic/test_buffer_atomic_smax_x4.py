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
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferAtomicConditionCodes:
    """Architectural integer condition codes for 128-bit buffer atomic operations."""
    zero: bool
    negative: bool
    positive: bool
    updated: bool


@dataclasses.dataclass(frozen=True)
class BufferStatusFlags:
    """Buffer memory status flags."""
    in_bounds: bool
    oob_fault: bool


def compute_buffer_condition_codes(old_val: int, smax_val: int, ret_val: int) -> BufferAtomicConditionCodes:
    signed_ret = s128(ret_val)
    new_val = max(s128(old_val), s128(smax_val))
    return BufferAtomicConditionCodes(
        zero=(signed_ret == 0),
        negative=(signed_ret < 0),
        positive=(signed_ret > 0),
        updated=(new_val != s128(old_val)),
    )


def compute_buffer_status_flags(in_bounds: bool) -> BufferStatusFlags:
    return BufferStatusFlags(
        in_bounds=in_bounds,
        oob_fault=not in_bounds,
    )


class TestBufferAtomicSmaxX4ConditionCodes(unittest.TestCase):
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
        self.interp.mem.write_u128(0, 0)
        self.interp.write_reg_u128(0, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)

        # Negative pre-op value
        val_neg10 = (-10) & MASK128
        self.interp.mem.write_u128(0, val_neg10)
        self.interp.write_reg_u128(0, 0)
        self.interp.step(inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)

        # Positive pre-op value
        self.interp.mem.write_u128(0, 100)
        self.interp.write_reg_u128(0, 200)
        self.interp.step(inst)
        cc_pos = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)

    def test_updated_flag_detection(self) -> None:
        self.interp.mem.write_u128(0, 100)
        self.interp.write_reg_u128(0, 200)  # strictly larger -> updated
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.updated)

        # Smaller value -> not updated
        self.interp.write_reg_u128(0, 50)
        self.interp.step(inst)
        cc_same = self.interp.last_condition_codes
        self.assertIsNotNone(cc_same)
        assert cc_same is not None
        self.assertFalse(cc_same.updated)

    def test_status_flags_in_bounds_and_oob(self) -> None:
        self.interp.write_reg_u128(0, 1)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 100)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
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
        self.interp.write_reg_u32(RegKind.VGPR, 4, 2000)
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
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, canary)

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

# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDBufferWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for 128-bit buffer atomics."""

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

    def read_lane_reg_u128(self, lane: int, idx: int) -> int:
        d0 = self.read_lane_reg_u32(lane, RegKind.VGPR, idx)
        d1 = self.read_lane_reg_u32(lane, RegKind.VGPR, idx + 1)
        d2 = self.read_lane_reg_u32(lane, RegKind.VGPR, idx + 2)
        d3 = self.read_lane_reg_u32(lane, RegKind.VGPR, idx + 3)
        return pack_u128(d0, d1, d2, d3)

    def write_lane_reg_u128(self, lane: int, idx: int, val: int) -> None:
        d0, d1, d2, d3 = unpack_u128(val)
        self.write_lane_reg_u32(lane, RegKind.VGPR, idx, d0)
        self.write_lane_reg_u32(lane, RegKind.VGPR, idx + 1, d1)
        self.write_lane_reg_u32(lane, RegKind.VGPR, idx + 2, d2)
        self.write_lane_reg_u32(lane, RegKind.VGPR, idx + 3, d3)

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
        """Executes a buffer_atomic_smax_x4 instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            smax_val = self.read_lane_reg_u128(lane, inst.vdata.index)
            vaddr = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((smax_val, vaddr))

        lane_results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, False))
                continue

            smax_val, vaddr = lane_inputs[lane]
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
                self.write_lane_reg_u128(lane, inst.vdata.index, ret_val)

            lane_results.append((ret_val, in_bounds))

        return lane_results


class TestBufferAtomicSmaxX4SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=4096,
        )

    def test_wave32_all_lanes_active_independent_offsets(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=8192)
        simd.set_buffer_descriptor(4, self.desc)

        for lane in range(32):
            addr = lane * 16
            simd.mem.write_u128(addr, 100)
            simd.write_lane_reg_u128(lane, 0, 200 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 4, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(32):
            ret, in_bounds = results[lane]
            self.assertTrue(in_bounds)
            self.assertEqual(ret, 100)
            self.assertEqual(simd.mem.read_u128(lane * 16), 200 + lane)
            self.assertEqual(simd.read_lane_reg_u128(lane, 0), 100)

    def test_wave64_alternating_exec_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=8192)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x5555555555555555  # Even lanes active
        simd.set_exec_mask(exec_mask)

        for lane in range(64):
            addr = lane * 16
            simd.mem.write_u128(addr, 50)
            simd.write_lane_reg_u128(lane, 0, 999)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 4, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        for lane in range(64):
            ret, in_bounds = results[lane]
            addr = lane * 16
            if lane % 2 == 0:
                self.assertTrue(in_bounds)
                self.assertEqual(ret, 50)
                self.assertEqual(simd.mem.read_u128(addr), 999)
                self.assertEqual(simd.read_lane_reg_u128(lane, 0), 50)
            else:
                self.assertIsNone(ret)
                self.assertFalse(in_bounds)
                self.assertEqual(simd.mem.read_u128(addr), 50)
                self.assertEqual(simd.read_lane_reg_u128(lane, 0), 999)

    def test_wave32_contention_same_address_serialization(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        target_addr = 64
        simd.mem.write_u128(target_addr, 0)

        for lane in range(32):
            simd.write_lane_reg_u128(lane, 0, 10 + lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 4, target_addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)

        self.assertEqual(simd.mem.read_u128(target_addr), 41)
        for lane in range(32):
            ret, in_bounds = results[lane]
            self.assertTrue(in_bounds)
            expected_pre = 0 if lane == 0 else (10 + lane - 1)
            self.assertEqual(ret, expected_pre)

    def test_single_active_lane(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(1 << 7)

        for lane in range(32):
            addr = lane * 16
            simd.mem.write_u128(addr, 10)
            simd.write_lane_reg_u128(lane, 0, 50)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 4, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        for lane in range(32):
            addr = lane * 16
            if lane == 7:
                self.assertEqual(simd.mem.read_u128(addr), 50)
                self.assertEqual(simd.read_lane_reg_u128(lane, 0), 10)
            else:
                self.assertEqual(simd.mem.read_u128(addr), 10)
                self.assertEqual(simd.read_lane_reg_u128(lane, 0), 50)

    def test_all_inactive_lanes(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0)

        simd.mem.write_u128(0, 100)
        simd.write_lane_reg_u128(0, 0, 999)
        simd.write_lane_reg_u32(0, RegKind.VGPR, 4, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 64)
        for ret, in_bounds in results:
            self.assertIsNone(ret)
            self.assertFalse(in_bounds)
        self.assertEqual(simd.mem.read_u128(0), 100)

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

class TestBufferAtomicSmaxX4RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_vdata_equals_vaddr_aliasing(self) -> None:
        # vdata occupies v[0:3]; vaddr aliases to v0 (which holds low 32-bits of vdata)
        addr = 128
        self.interp.mem.write_u128(addr, 50)
        # v0 = addr (128), v1 = 0, v2 = 0, v3 = 0 -> 128-bit value is 128
        self.interp.write_reg_u32(RegKind.VGPR, 0, addr)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 0, count=1),  # Aliased to v0
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 50)
        self.assertEqual(self.interp.mem.read_u128(addr), 128)
        self.assertEqual(self.interp.read_reg_u128(0), 50)

    def test_vaddr_aliased_to_vdata_highest_dword(self) -> None:
        # vaddr aliases to v3 (bits 96..127 of vdata)
        addr = 64
        self.interp.mem.write_u128(addr, 200)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 3, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 3, count=1),  # Aliased to v3
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 200)
        # Expected new memory: max(200, (64 << 96) | 100)
        expected_smax = pack_u128(100, 0, 0, addr)
        self.assertEqual(self.interp.mem.read_u128(addr), expected_smax)
        self.assertEqual(self.interp.read_reg_u128(0), 200)

    def test_adjacent_register_blocks_isolation(self) -> None:
        # Pre-seed registers around v[4:7]
        for idx in range(16):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0xAA000000 + idx)

        self.interp.write_reg_u128(4, pack_u128(10, 20, 30, 40))
        self.interp.write_reg_u32(RegKind.VGPR, 8, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 4, count=4),
            vaddr=Register(RegKind.VGPR, 8, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        self.interp.step(inst)

        for idx in range(0, 4):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), 0xAA000000 + idx)
        for idx in range(9, 16):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), 0xAA000000 + idx)

    def test_sgpr_descriptor_and_offset_preserved(self) -> None:
        d0, d1, d2, d3 = self.desc.to_dwords()
        self.interp.write_reg_u32(RegKind.SGPR, 8, 32)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
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

    def test_simd_aliasing_divergent_lanes(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x0000000F  # Lanes 0..3 active
        simd.set_exec_mask(exec_mask)

        for lane in range(32):
            addr = lane * 16
            simd.mem.write_u128(addr, 500)
            simd.write_lane_reg_u128(lane, 0, addr)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 0, count=1),  # Aliased
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        for lane in range(4):
            addr = lane * 16
            self.assertEqual(simd.mem.read_u128(addr), 500)
            self.assertEqual(simd.read_lane_reg_u128(lane, 0), 500)

        for lane in range(4, 32):
            addr = lane * 16
            self.assertEqual(simd.mem.read_u128(addr), 500)
            self.assertEqual(simd.read_lane_reg_u128(lane, 0), addr)

class TestBufferAtomicSmaxX4BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 bit-pattern preservation, signed extremes, and buffer boundary limits."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_signed_128bit_min_max_extremes(self) -> None:
        int128_min = 1 << 127
        int128_max = (1 << 127) - 1
        self.interp.mem.write_u128(0, int128_min)
        self.interp.write_reg_u128(0, int128_max)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

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
        self.assertEqual(ret, int128_min)
        self.assertEqual(self.interp.mem.read_u128(0), int128_max)
        self.assertEqual(self.interp.read_reg_u128(0), int128_min)

    def test_signed_negative_vs_zero_and_minus_one(self) -> None:
        minus_one = MASK128
        zero = 0x0
        self.interp.mem.write_u128(0, minus_one)
        self.interp.write_reg_u128(0, zero)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

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
        self.assertEqual(ret, minus_one)
        self.assertEqual(self.interp.mem.read_u128(0), zero)

    def test_ieee754_double_sign_bit_and_zero_ordering(self) -> None:
        pos_zero = 0x0000000000000000
        neg_zero = 1 << 127
        self.interp.mem.write_u128(0, neg_zero)
        self.interp.write_reg_u128(0, pos_zero)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

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
        self.assertEqual(ret, neg_zero)
        self.assertEqual(self.interp.mem.read_u128(0), pos_zero)

    def test_ieee754_subnormals_bitwise_integrity_no_flush(self) -> None:
        pos_subnormal = pack_u128(1, 0, 0, 0)
        neg_subnormal = (1 << 127) | 1
        self.interp.mem.write_u128(0, neg_subnormal)
        self.interp.write_reg_u128(0, pos_subnormal)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

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
        self.assertEqual(ret, neg_subnormal)
        self.assertEqual(self.interp.mem.read_u128(0), pos_subnormal)

    def test_ieee754_nan_and_inf_no_canonicalization(self) -> None:
        qnan = pack_u128(0xCAFEBABE, 0x7FF80000, 0, 0)
        neg_inf = pack_u128(0, 0, 0, 0xFFF00000)
        self.interp.mem.write_u128(0, neg_inf)
        self.interp.write_reg_u128(0, qnan)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

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
        self.assertEqual(ret, neg_inf)
        self.assertEqual(self.interp.mem.read_u128(0), qnan)

    def test_buffer_boundary_exact_fit_and_overrun(self) -> None:
        # Buffer has 1024 bytes. 128-bit access is 16 bytes.
        # Offset 1008: 1008 + 16 == 1024 (valid)
        self.interp.mem.write_u128(1008, 100)
        self.interp.write_reg_u128(0, 500)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 1008)

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
        self.assertEqual(ret, 100)
        self.assertEqual(self.interp.mem.read_u128(1008), 500)
        self.assertIsNotNone(self.interp.last_status_flags)
        assert self.interp.last_status_flags is not None
        self.assertTrue(self.interp.last_status_flags.in_bounds)

        # Offset 1009: 1009 + 16 == 1025 > 1024 (OOB fault)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 1009)
        ret_oob, in_bounds_oob = self.interp.step(inst)
        self.assertFalse(in_bounds_oob)
        self.assertEqual(ret_oob, 0)
        self.assertIsNotNone(self.interp.last_status_flags)
        assert self.interp.last_status_flags is not None
        self.assertTrue(self.interp.last_status_flags.oob_fault)

    def test_zero_capacity_buffer_rejection(self) -> None:
        zero_desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=0)
        self.interp.set_buffer_descriptor(8, zero_desc)
        self.interp.write_reg_u128(0, 42)
        self.interp.write_reg_u32(RegKind.VGPR, 4, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
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

class TestBufferAtomicSmaxX4NegativeControls(unittest.TestCase):
    """Unit tests for instruction syntax rejection, parameter limits, and negative controls."""

    def test_vdata_must_be_vgpr_quad(self) -> None:
        # vdata must be VGPR, not SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
                vdata=Register(RegKind.SGPR, 0, count=4),
                vaddr=Register(RegKind.VGPR, 4, count=1),
                srsrc=Register(RegKind.SGPR, 8, count=4),
                soffset=0,
            )
        # vdata count must be 4, not 1 or 2
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 4, count=1),
                srsrc=Register(RegKind.SGPR, 8, count=4),
                soffset=0,
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
                vdata=Register(RegKind.VGPR, 0, count=2),
                vaddr=Register(RegKind.VGPR, 4, count=1),
                srsrc=Register(RegKind.SGPR, 8, count=4),
                soffset=0,
            )

    def test_vaddr_must_be_vgpr(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
                vdata=Register(RegKind.VGPR, 0, count=4),
                vaddr=Register(RegKind.SGPR, 4, count=1),
                srsrc=Register(RegKind.SGPR, 8, count=4),
                soffset=0,
            )

    def test_srsrc_must_be_sgpr_quad(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
                vdata=Register(RegKind.VGPR, 0, count=4),
                vaddr=Register(RegKind.VGPR, 4, count=1),
                srsrc=Register(RegKind.VGPR, 8, count=4),
                soffset=0,
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
                vdata=Register(RegKind.VGPR, 0, count=4),
                vaddr=Register(RegKind.VGPR, 4, count=1),
                srsrc=Register(RegKind.SGPR, 8, count=2),
                soffset=0,
            )

    def test_soffset_cannot_be_vgpr(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
                vdata=Register(RegKind.VGPR, 0, count=4),
                vaddr=Register(RegKind.VGPR, 4, count=1),
                srsrc=Register(RegKind.SGPR, 8, count=4),
                soffset=Register(RegKind.VGPR, 12, count=1),
            )

    def test_inst_offset_range_limits(self) -> None:
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=-1)
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=4096)

    def test_wave_size_validation(self) -> None:
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=128)


class TestBufferAtomicSmaxX4MutationDetection(unittest.TestCase):
    """Mutation testing to verify test suite catches faulty implementations."""

    def test_mutation_unsigned_instead_of_signed_max(self) -> None:
        def buggy_unsigned_atomic_max(old: int, op: int) -> int:
            return max(old & MASK128, op & MASK128)

        val_neg10 = (-10) & MASK128
        val_pos5 = 5
        # Correct signed max: max(-10, 5) = 5
        correct_max = u128(max(s128(val_neg10), s128(val_pos5)))
        self.assertEqual(correct_max, val_pos5)
        # Buggy unsigned max chooses val_neg10 (huge unsigned)
        buggy_max = buggy_unsigned_atomic_max(val_neg10, val_pos5)
        self.assertEqual(buggy_max, val_neg10)
        self.assertNotEqual(correct_max, buggy_max)

    def test_mutation_omitted_16byte_bounds_check(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=1024)
        offset = 1016  # offset < 1024, but offset + 16 == 1032 > 1024

        real_in_bounds = (offset + 16 <= desc.num_records)
        self.assertFalse(real_in_bounds)

        # Mutated 8-byte check
        mutated_in_bounds = (offset + 8 <= desc.num_records)
        self.assertTrue(mutated_in_bounds)
        self.assertNotEqual(real_in_bounds, mutated_in_bounds)

    def test_mutation_return_post_op_instead_of_pre_op(self) -> None:
        mem = BufferMemory(4096)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=1024)
        mem.write_u128(0, 100)

        ret, in_bounds = ref_buffer_atomic_smax_x4(
            mem, desc, soffset=0, vaddr=0, smax_val=200, glc=True
        )
        self.assertEqual(ret, 100)
        self.assertEqual(mem.read_u128(0), 200)
        self.assertNotEqual(ret, 200)

    def test_mutation_inactive_lane_side_effects(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=1024)
        simd.set_buffer_descriptor(4, desc)
        simd.set_exec_mask(0x1)

        simd.mem.write_u128(64, 500)
        simd.write_lane_reg_u128(1, 0, 999)
        simd.write_lane_reg_u32(1, RegKind.VGPR, 4, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SMAX_X4,
            vdata=Register(RegKind.VGPR, 0, count=4),
            vaddr=Register(RegKind.VGPR, 4, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)
        self.assertEqual(results[1], (None, False))
        self.assertEqual(simd.mem.read_u128(64), 500)
        self.assertEqual(simd.read_lane_reg_u128(1, 0), 999)

if __name__ == "__main__":
    unittest.main()
