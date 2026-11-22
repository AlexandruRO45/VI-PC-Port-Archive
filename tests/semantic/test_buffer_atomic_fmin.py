"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_fmin.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory 32-bit floating point atomic minimum:
  buffer_atomic_fmin vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the IEEE 754 32-bit float value. If 'glc' is enabled,
       the pre-operation 32-bit float value at memory is returned to vdata.
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
       The atomic operation is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Float Min Operation (IEEE 754-2008 minNum):
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = min_f32(old_val, vdata)
       memory[effective_address] = new_val
       if glc: vdata = old_val
     Rules:
       - If one operand is NaN, the other (valid number) is returned.
       - If both operands are NaN, canonical quiet NaN (0x7FC00000) is returned.
       - Signed zeroes: -0.0 is strictly less than +0.0 (-0.0 < +0.0).
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
EXP_MASK: int = 0x7F800000
MANT_MASK: int = 0x007FFFFF
SIGN_MASK: int = 0x80000000

POS_ZERO: int = 0x00000000
NEG_ZERO: int = 0x80000000
POS_INF: int = 0x7F800000
NEG_INF: int = 0xFF800000
QNAN: int = 0x7FC00000
SNAN: int = 0x7F800001


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_bits(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 NaN."""
    bits &= MASK32
    return (bits & EXP_MASK) == EXP_MASK and (bits & MANT_MASK) != 0


def zero_ordered_key(v: float) -> tuple[float, float]:
    """Sort key ordering -0.0 strictly below +0.0."""
    return (v, math.copysign(1.0, v))


def ref_min_f32(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 float min (IEEE 754-2008 minNum).

    Rules:
      - If one operand is NaN, return the other operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FC00000).
      - If values are non-NaN, order -0.0 < +0.0.
    """
    s0 &= MASK32
    s1 &= MASK32
    nan0 = is_nan_bits(s0)
    nan1 = is_nan_bits(s1)
    if nan0 and nan1:
        return QNAN
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f32(s0)
    f1 = bits_to_f32(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 <= k1 else s1


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


def ref_buffer_atomic_fmin(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    fmin_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_fmin.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (fmin_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = ref_min_f32(old_val, fmin_val & MASK32)
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (fmin_val & MASK32)
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
    BUFFER_ATOMIC_FMIN = "buffer_atomic_fmin"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_fmin instruction."""
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


class TestBufferAtomicFminAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0, count=1)
        v1 = Register(RegKind.VGPR, 1, count=1)
        s_desc = Register(RegKind.SGPR, 4, count=4)
        s_off = Register(RegKind.SGPR, 8, count=1)

        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v1.format_asm(), "v1")
        self.assertEqual(s_desc.format_asm(), "s[4:7]")
        self.assertEqual(s_off.format_asm(), "s8")

    def test_instruction_formatting(self) -> None:
        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FMIN,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_fmin v0, v1, s[4:7], s8 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FMIN,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 3, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_fmin v2, v3, s[12:15], 0 offen",
        )

    def test_instruction_validation(self) -> None:
        # Invalid vdata kind (must be VGPR)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FMIN,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        # Invalid srsrc count (must be 4)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FMIN,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

        # Invalid offset (> 4095)
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=5000)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FMIN,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        ))
        expected = "buffer_atomic_fmin v0, v1, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


class TestBufferAtomicFminOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_fmin_in_bounds(self) -> None:
        # Pre-seed address 100 with float 10.0
        self.mem.write_u32(100, f32_to_bits(10.0))
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=0, vaddr=100, fmin_val=f32_to_bits(5.0), glc=False
        )
        self.assertTrue(in_bounds)
        # Memory should now hold min(10.0, 5.0) = 5.0
        self.assertEqual(self.mem.read_u32(100), f32_to_bits(5.0))

    def test_atomic_fmin_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(200, f32_to_bits(25.0))
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=0, vaddr=200, fmin_val=f32_to_bits(30.0), glc=True
        )
        self.assertTrue(in_bounds)
        # glc returns old value (25.0)
        self.assertEqual(ret, f32_to_bits(25.0))
        # memory retains min(25.0, 30.0) = 25.0
        self.assertEqual(self.mem.read_u32(200), f32_to_bits(25.0))

    def test_out_of_bounds_rejection(self) -> None:
        # num_records is 1024; offset 1024 is out of bounds
        self.mem.write_u32(1024, f32_to_bits(1.0))
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=0, vaddr=1024, fmin_val=f32_to_bits(-10.0), glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), f32_to_bits(1.0))

    def test_negative_zero_strictly_less_than_positive_zero(self) -> None:
        # -0.0 < +0.0 in IEEE 754-2008 and RDNA2
        self.mem.write_u32(0, POS_ZERO)
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=0, vaddr=0, fmin_val=NEG_ZERO, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, POS_ZERO)
        self.assertEqual(self.mem.read_u32(0), NEG_ZERO)

    def test_nan_suppression_rule(self) -> None:
        # min(number, NaN) = number; min(NaN, number) = number
        self.mem.write_u32(16, f32_to_bits(42.0))
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=0, vaddr=16, fmin_val=QNAN, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, f32_to_bits(42.0))
        self.assertEqual(self.mem.read_u32(16), f32_to_bits(42.0))

        # Memory has NaN, input has number
        self.mem.write_u32(20, QNAN)
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=0, vaddr=20, fmin_val=f32_to_bits(42.0), glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, QNAN)
        self.assertEqual(self.mem.read_u32(20), f32_to_bits(42.0))

    def test_both_nan_returns_canonical_qnan(self) -> None:
        self.mem.write_u32(24, SNAN)
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=0, vaddr=24, fmin_val=QNAN, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, SNAN)
        self.assertEqual(self.mem.read_u32(24), QNAN)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(64) + inst_offset(16) + vaddr(20) = 100
        self.mem.write_u32(100, f32_to_bits(20.0))
        ret, in_bounds = ref_buffer_atomic_fmin(
            self.mem, self.desc, soffset=64, vaddr=20, fmin_val=f32_to_bits(15.0), inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, f32_to_bits(20.0))
        self.assertEqual(self.mem.read_u32(100), f32_to_bits(15.0))

    def test_descriptor_to_from_dwords_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x00007FFF12340000,
            stride=64,
            num_records=65536,
            format_flags=0x12345678,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc.base_address, restored.base_address)
        self.assertEqual(desc.stride, restored.stride)
        self.assertEqual(desc.num_records, restored.num_records)
        self.assertEqual(desc.format_flags, restored.format_flags)


if __name__ == "__main__":
    unittest.main()
