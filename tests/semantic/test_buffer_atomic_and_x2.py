from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64 = 1 << 63


# ---------------------------------------------------------------------------
# Reference Mathematical Oracle and Architectural Model
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferResourceDescriptor:
    """RDNA2 128-bit Buffer Resource Descriptor (V# / SQ_BUF_RSRC_WORD0..3)."""
    base_address: int
    stride: int
    num_records: int
    format_flags: int = 0

    @classmethod
    def from_dwords(cls, d0: int, d1: int, d2: int, d3: int) -> BufferResourceDescriptor:
        base = (d0 & MASK32) | ((d1 & 0xFFFF) << 32)
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


def ref_buffer_atomic_and_x2(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    and_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_and_x2.

    Computes 64-bit untyped buffer atomic bitwise AND:
      mem[addr] = mem[addr] & and_val

    Returns:
      (return_val, was_in_bounds)
      where return_val is the pre-operation memory value if glc=True, else 0.
    """
    effective_offset = (vaddr if offen else 0) + soffset + inst_offset
    if effective_offset < 0 or effective_offset + 8 > desc.num_records:
        return 0, False

    phys_addr = desc.base_address + effective_offset
    prev_val = mem.read_u64(phys_addr)
    new_val = (prev_val & and_val) & MASK64
    mem.write_u64(phys_addr, new_val)

    ret_val = prev_val if glc else 0
    return ret_val, True


class TestBufferAtomicAndX2Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and memory models."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=4096,
        )

    def test_oracle_basic_and_with_glc(self) -> None:
        self.mem.write_u64(64, 0xFF00FF00FF00FF00)
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=64, and_val=0x0F0F0F0F0F0F0F0F, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xFF00FF00FF00FF00)
        self.assertEqual(self.mem.read_u64(64), 0x0F000F000F000F00)

    def test_oracle_glc_false_returns_zero(self) -> None:
        self.mem.write_u64(64, 0x1234567890ABCDEF)
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=64, and_val=0x00000000FFFFFFFF, glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(64), 0x0000000090ABCDEF)

    def test_oracle_out_of_bounds_high(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=100)
        self.mem.write_u64(96, 0xAAAAAAAAAAAAAAAA)
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, desc, soffset=0, vaddr=96, and_val=0x5555555555555555, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(96), 0xAAAAAAAAAAAAAAAA)

    def test_oracle_out_of_bounds_negative(self) -> None:
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=-8, vaddr=0, and_val=0xFF, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)

    def test_oracle_descriptor_dwords_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x123456789000,
            stride=64,
            num_records=16384,
            format_flags=0x80000000,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc.base_address, restored.base_address)
        self.assertEqual(desc.stride, restored.stride)
        self.assertEqual(desc.num_records, restored.num_records)
        self.assertEqual(desc.format_flags, restored.format_flags)

    def test_oracle_all_ones_identity_and_zero_annihilation(self) -> None:
        self.mem.write_u64(128, 0xDEADBEEFCAFEBABE)
        # Bitwise AND with all 1s preserves value
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=128, and_val=MASK64, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEFCAFEBABE)
        self.assertEqual(self.mem.read_u64(128), 0xDEADBEEFCAFEBABE)

        # Bitwise AND with 0 clears to 0
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=128, and_val=0, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEFCAFEBABE)
        self.assertEqual(self.mem.read_u64(128), 0)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register kinds in RDNA2 architecture."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Register operand with kind, index, and optional multi-register grouping."""
    kind: RegKind
    index: int
    count: int = 1

    def format_asm(self) -> str:
        if self.count == 1:
            return f"{self.kind.value}{self.index}"
        return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR


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
    """64-bit buffer atomic opcodes."""
    BUFFER_ATOMIC_AND_X2 = "buffer_atomic_and_x2"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_and_x2 instruction."""
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


class TestBufferAtomicAndX2AST(unittest.TestCase):
    """Unit tests for AST data structures and assembly emission."""

    def test_ast_instruction_creation(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
        )
        self.assertEqual(inst.opcode, Opcode.BUFFER_ATOMIC_AND_X2)
        self.assertEqual(inst.vdata.format_asm(), "v[0:1]")
        self.assertEqual(inst.vaddr.format_asm(), "v2")
        self.assertEqual(inst.srsrc.format_asm(), "s[0:3]")
        self.assertEqual(inst.to_assembly(), "buffer_atomic_and_x2 v[0:1], v2, s[0:3], 0 offen")

    def test_ast_to_assembly_with_modifiers(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 4, count=2),
            vaddr=Register(RegKind.VGPR, 6),
            srsrc=Register(RegKind.SGPR, 8, count=4),
            soffset=Register(RegKind.SGPR, 12),
            mods=BufferAtomicModifiers(offset=64, glc=True, slc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_and_x2 v[4:5], v6, s[8:11], s12 offen offset:64 glc slc",
        )

    def test_ast_program_assembly(self) -> None:
        prog = Program()
        inst1 = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
        )
        inst2 = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 2, count=2),
            vaddr=Register(RegKind.VGPR, 4),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=16,
            mods=BufferAtomicModifiers(glc=True),
        )
        prog.append(inst1)
        prog.append(inst2)
        asm = prog.to_assembly()
        lines = asm.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("buffer_atomic_and_x2 v[0:1]"))
        self.assertTrue(lines[1].endswith("glc"))


if __name__ == "__main__":
    unittest.main()
