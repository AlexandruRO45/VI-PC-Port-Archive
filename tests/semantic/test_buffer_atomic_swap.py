"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_swap.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory atomic swap:
  buffer_atomic_swap vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit dword value to swap into memory.
       If 'glc' (Globally Coherent) is enabled, the pre-operation dword value
       at memory is returned to vdata.
     - vaddr: VGPR containing the byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = soffset + inst_offset + (vaddr if offen else 0)
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic swap is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = vdata
       memory[effective_address] = new_val
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
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


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


def ref_buffer_atomic_swap(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    swap_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_swap.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (swap_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = swap_val & MASK32
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (swap_val & MASK32)
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
    BUFFER_ATOMIC_SWAP = "buffer_atomic_swap"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_swap instruction."""
    opcode: Opcode
    vdata: Register
    vaddr: Register
    srsrc: Register
    soffset: Register | int
    mods: BufferAtomicModifiers = dataclasses.field(default_factory=BufferAtomicModifiers)

    def __post_init__(self) -> None:
        if not self.vdata.is_vgpr or self.vdata.count != 1:
            raise ValueError(f"vdata must be a single 32-bit VGPR: {self.vdata}")
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
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_swap."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.vcc: int = 0
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
        """Executes a single buffer_atomic_swap instruction."""
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            swap_val=vdata,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u32(RegKind.VGPR, inst.vdata.index, ret_val)

        self.trace.append(f"{inst.to_assembly()} => ret={ret_val:#010x}, in_bounds={in_bounds}")
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestBufferAtomicSwapOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_swap_in_bounds(self) -> None:
        self.mem.write_u32(64, 0x12345678)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            swap_val=0x9ABCDEF0,
            inst_offset=0,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0x9ABCDEF0)
        self.assertEqual(self.mem.read_u32(64), 0x9ABCDEF0)

    def test_atomic_swap_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(128, 0xCAFEBABE)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            swap_val=0xDEADBEEF,
            inst_offset=0,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0xCAFEBABE)
        self.assertEqual(self.mem.read_u32(128), 0xDEADBEEF)

    def test_out_of_bounds_rejection(self) -> None:
        self.mem.write_u32(1020, 0x11111111)
        # 1024 + 4 > 1024 (NUM_RECORDS) -> OOB
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1024,
            swap_val=0x99999999,
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret_val, 0)
        # Memory must remain untouched
        self.assertEqual(self.mem.read_u32(1020), 0x11111111)

    def test_swap_identical_value(self) -> None:
        self.mem.write_u32(32, 0x55555555)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=32,
            swap_val=0x55555555,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0x55555555)
        self.assertEqual(self.mem.read_u32(32), 0x55555555)

    def test_inst_offset_and_soffset_combination(self) -> None:
        self.mem.write_u32(256 + 16 + 8, 0x42)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=256,
            vaddr=16,
            swap_val=0x1337,
            inst_offset=8,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0x42)
        self.assertEqual(self.mem.read_u32(256 + 16 + 8), 0x1337)


class TestBufferAtomicSwapAST(unittest.TestCase):
    """Unit tests for symbolic AST representation and formatting."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertTrue(v0.is_vgpr)
        self.assertFalse(v0.is_sgpr)

        s_desc = Register(RegKind.SGPR, 8, count=4)
        self.assertEqual(s_desc.format_asm(), "s[8:11]")
        self.assertTrue(s_desc.is_sgpr)

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 0, count=3)

    def test_instruction_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SWAP,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 2),
            mods=BufferAtomicModifiers(offen=True, offset=16, glc=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_swap v0, v1, s[4:7], s2 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SWAP,
            vdata=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 6),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=False, slc=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_swap v5, v6, s[0:3], 0 slc",
        )

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SWAP,
                vdata=Register(RegKind.SGPR, 0),  # must be VGPR
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SWAP,
                vdata=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 4, count=2),  # count must be 4
                soffset=0,
            )

        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SWAP,
                vdata=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=Register(RegKind.VGPR, 2),  # soffset must be SGPR or int
            )

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SWAP,
                vdata=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=True),
            )
        )
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SWAP,
                vdata=Register(RegKind.VGPR, 2),
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=False),
            )
        )
        lines = prog.to_assembly().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("glc", lines[0])
        self.assertNotIn("glc", lines[1])


class TestBufferAtomicSwapScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_atomic_swap_execution(self) -> None:
        # Pre-seed memory at byte offset 64 with 0xAAAAAAAA
        self.interp.mem.write_u32(64, 0xAAAAAAAA)
        # v0 = new swap value (0xBBBBBBBB), v1 = offset (64)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0xBBBBBBBB)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SWAP,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )

        ret_val, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0xAAAAAAAA)
        # v0 must have received pre-op memory value
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0xAAAAAAAA)
        # Memory must now hold the swapped value
        self.assertEqual(self.interp.mem.read_u32(64), 0xBBBBBBBB)

    def test_scalar_atomic_swap_without_glc_preserves_vgpr(self) -> None:
        self.interp.mem.write_u32(100, 0x11112222)
        self.interp.write_reg_u32(RegKind.VGPR, 5, 0x33334444)
        self.interp.write_reg_u32(RegKind.VGPR, 6, 100)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SWAP,
            vdata=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 6),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )

        ret_val, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # Without glc, v5 retains its original value
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 5), 0x33334444)
        # Memory holds the new swapped value
        self.assertEqual(self.interp.mem.read_u32(100), 0x33334444)

    def test_scalar_oob_suppression(self) -> None:
        self.interp.mem.write_u32(1020, 0x99999999)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x77777777)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 2048)  # OOB

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_SWAP,
            vdata=Register(RegKind.VGPR, 0),
            vaddr=Register(RegKind.VGPR, 1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )

        ret_val, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertEqual(ret_val, 0)
        # v0 receives 0 on OOB with glc
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0)
        # Memory untouched
        self.assertEqual(self.interp.mem.read_u32(1020), 0x99999999)

    def test_scalar_program_chain(self) -> None:
        # Swap sequentially at offset 0: 0x100 -> 0x200 -> 0x300
        self.interp.mem.write_u32(0, 0x100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x200)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        prog = Program()
        prog.append(
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_SWAP,
                vdata=Register(RegKind.VGPR, 0),
                vaddr=Register(RegKind.VGPR, 1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
                mods=BufferAtomicModifiers(glc=True),
            )
        )

        results = self.interp.run_program(prog)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 0x100)
        self.assertEqual(self.interp.mem.read_u32(0), 0x200)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0x100)


if __name__ == "__main__":
    unittest.main()
