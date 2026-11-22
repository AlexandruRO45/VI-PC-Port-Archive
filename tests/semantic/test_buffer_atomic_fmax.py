"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_fmax.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory single-precision float atomic maximum:
  buffer_atomic_fmax vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit single-precision float (f32) value to compare atomically.
       If 'glc' is enabled, the pre-operation dword value at memory is returned to vdata.
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
       The atomic maximum is suppressed (memory is NOT modified).
       If 'glc' is set, 0 (0x00000000) is returned to vdata.
  3. Atomic Operation:
     For in-bounds accesses:
       old_bits = memory[effective_address]
       new_bits = fmax(old_bits, vdata)
       memory[effective_address] = new_bits
       if glc: vdata = old_bits
  4. IEEE 754 Float Max Ordering Rules:
     - NaN suppression: if one operand is NaN, return the non-NaN operand.
     - If both operands are NaN, return canonical quiet NaN (0x7FC00000).
     - Signed zero ordering: -0.0 < +0.0.
  5. Inactive Lanes (EXEC mask):
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

SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def bits_to_f32(bits: int) -> float:
    """Reinterpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Reinterpret IEEE 754 single-precision float as 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_nan_f32(bits: int) -> bool:
    """Checks if 32-bit pattern is any IEEE 754 NaN."""
    return (bits & EXP_MASK_F32) == EXP_MASK_F32 and (bits & MANT_MASK_F32) != 0


def is_qnan_f32(bits: int) -> bool:
    """Checks if 32-bit pattern is a quiet NaN."""
    return is_nan_f32(bits) and bool(bits & 0x00400000)


def is_snan_f32(bits: int) -> bool:
    """Checks if 32-bit pattern is a signaling NaN."""
    return is_nan_f32(bits) and not bool(bits & 0x00400000)


def zero_ordered_key(v: float) -> tuple[float, float]:
    """Tuple key that orders -0.0 strictly less than +0.0."""
    return (v, math.copysign(1.0, v))


def ref_fmax_f32(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 32-bit float max.

    Rules:
      - If one operand is NaN, return the non-NaN operand unchanged.
      - If both operands are NaN, return canonical quiet NaN (0x7FC00000).
      - If non-NaN values, order -0.0 < +0.0.
    """
    s0 &= MASK32
    s1 &= MASK32
    nan0 = is_nan_f32(s0)
    nan1 = is_nan_f32(s1)
    if nan0 and nan1:
        return QNAN_F32
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f32(s0)
    f1 = bits_to_f32(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 >= k1 else s1


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


def ref_buffer_atomic_fmax(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    comp_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_fmax.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        return (0 if glc else (comp_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & 0xFFFFFFFFFFFFFFFF
    old_bits = mem.read_u32(phys_addr)
    # Float maximum comparison
    new_bits = ref_fmax_f32(old_bits, comp_val & MASK32)
    mem.write_u32(phys_addr, new_bits)

    ret_val = old_bits if glc else (comp_val & MASK32)
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
    BUFFER_ATOMIC_FMAX = "buffer_atomic_fmax"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_fmax instruction."""
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


class TestBufferAtomicFmaxOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_fmax_in_bounds(self) -> None:
        self.mem.write_u32(100, f32_to_bits(1.5))
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=100, comp_val=f32_to_bits(3.5), glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f32(self.mem.read_u32(100)), 3.5)

    def test_atomic_fmax_smaller_value_no_change(self) -> None:
        self.mem.write_u32(100, f32_to_bits(10.0))
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=100, comp_val=f32_to_bits(5.0), glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f32(self.mem.read_u32(100)), 10.0)

    def test_atomic_fmax_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(200, f32_to_bits(2.0))
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=200, comp_val=f32_to_bits(8.0), glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f32(ret), 2.0)
        self.assertEqual(bits_to_f32(self.mem.read_u32(200)), 8.0)

    def test_atomic_fmax_signed_zeros(self) -> None:
        # -0.0 < +0.0
        self.mem.write_u32(0, NEG_ZERO_F32)
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=0, comp_val=POS_ZERO_F32, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, NEG_ZERO_F32)
        self.assertEqual(self.mem.read_u32(0), POS_ZERO_F32)

        # +0.0 vs -0.0 -> stays +0.0
        ret2, in_bounds2 = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=0, comp_val=NEG_ZERO_F32, glc=True
        )
        self.assertTrue(in_bounds2)
        self.assertEqual(ret2, POS_ZERO_F32)
        self.assertEqual(self.mem.read_u32(0), POS_ZERO_F32)

    def test_atomic_fmax_infinities(self) -> None:
        # -inf vs 100.0 -> 100.0 wins
        self.mem.write_u32(4, NEG_INF_F32)
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=4, comp_val=f32_to_bits(100.0), glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, NEG_INF_F32)
        self.assertEqual(bits_to_f32(self.mem.read_u32(4)), 100.0)

        # +inf vs 1e30 -> +inf wins
        self.mem.write_u32(8, f32_to_bits(1e30))
        ret2, in_bounds2 = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=8, comp_val=POS_INF_F32, glc=True
        )
        self.assertTrue(in_bounds2)
        self.assertEqual(self.mem.read_u32(8), POS_INF_F32)

    def test_atomic_fmax_nan_suppression(self) -> None:
        # NaN in memory, valid float comp_val -> valid float wins
        self.mem.write_u32(12, QNAN_F32)
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=12, comp_val=f32_to_bits(42.0), glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, QNAN_F32)
        self.assertEqual(bits_to_f32(self.mem.read_u32(12)), 42.0)

        # Valid float in memory, NaN comp_val -> memory remains valid float
        ret2, in_bounds2 = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=12, comp_val=SNAN_F32, glc=True
        )
        self.assertTrue(in_bounds2)
        self.assertEqual(bits_to_f32(self.mem.read_u32(12)), 42.0)

        # Dual NaN -> produces canonical quiet NaN
        self.mem.write_u32(16, SNAN_F32)
        ret3, in_bounds3 = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=16, comp_val=QNAN_F32, glc=True
        )
        self.assertTrue(in_bounds3)
        self.assertEqual(self.mem.read_u32(16), QNAN_F32)

    def test_out_of_bounds_rejection(self) -> None:
        self.mem.write_u32(1024, f32_to_bits(50.0))
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=0, vaddr=1024, comp_val=f32_to_bits(100.0), glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(bits_to_f32(self.mem.read_u32(1024)), 50.0)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(64) + inst_offset(16) + vaddr(20) = 100
        self.mem.write_u32(100, f32_to_bits(10.0))
        ret, in_bounds = ref_buffer_atomic_fmax(
            self.mem, self.desc, soffset=64, vaddr=20, comp_val=f32_to_bits(50.0), inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(bits_to_f32(ret), 10.0)
        self.assertEqual(bits_to_f32(self.mem.read_u32(100)), 50.0)


class TestBufferAtomicFmaxAST(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_FMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_fmax v0, v1, s[4:7], s8 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FMAX,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 3, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_fmax v2, v3, s[12:15], 0 offen",
        )

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FMAX,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FMAX,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=5000)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FMAX,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        ))
        expected = "buffer_atomic_fmax v0, v1, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


if __name__ == "__main__":
    unittest.main()
