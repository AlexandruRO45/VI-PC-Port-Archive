"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_fadd.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory 32-bit floating point atomic addition:
  buffer_atomic_fadd vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

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
       The atomic float add is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation (IEEE 754-2008 single-precision addition):
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = ieee754_add_f32(old_val, vdata)
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
QNAN_DEFAULT: int = 0x7FC00000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_MASK) else val


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def u32_to_f32(val: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", u32(val)))[0]


def f32_to_u32(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def ieee754_add_f32(a_u32: int, b_u32: int) -> int:
    """Bit-accurate IEEE 754 single-precision addition matching RDNA2 hardware semantics."""
    a_u32 = u32(a_u32)
    b_u32 = u32(b_u32)

    a_exp = (a_u32 >> 23) & 0xFF
    a_mant = a_u32 & MANT_MASK
    b_exp = (b_u32 >> 23) & 0xFF
    b_mant = b_u32 & MANT_MASK

    a_is_nan = (a_exp == 0xFF and a_mant != 0)
    b_is_nan = (b_exp == 0xFF and b_mant != 0)

    # Any NaN operand produces a canonical quiet NaN
    if a_is_nan or b_is_nan:
        return QNAN_DEFAULT

    a_is_inf = (a_exp == 0xFF and a_mant == 0)
    b_is_inf = (b_exp == 0xFF and b_mant == 0)
    a_sign = (a_u32 >> 31) & 1
    b_sign = (b_u32 >> 31) & 1

    # (+Inf) + (-Inf) or (-Inf) + (+Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign != b_sign):
        return QNAN_DEFAULT

    # If either is Inf, result is that Inf
    if a_is_inf:
        return a_u32
    if b_is_inf:
        return b_u32

    # Handle IEEE -0.0 + -0.0 = -0.0
    if a_u32 == NEG_ZERO and b_u32 == NEG_ZERO:
        return NEG_ZERO

    a_f = u32_to_f32(a_u32)
    b_f = u32_to_f32(b_u32)

    try:
        res_f = a_f + b_f
    except OverflowError:
        return POS_INF if (a_f > 0 or b_f > 0) else NEG_INF

    res_u32 = f32_to_u32(res_f)
    # When sum is -0.0 but not both operands were -0.0, IEEE 754 rounds to +0.0 in nearest-even
    if res_u32 == NEG_ZERO and not (a_u32 == NEG_ZERO and b_u32 == NEG_ZERO):
        res_u32 = POS_ZERO

    return res_u32


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


def ref_buffer_atomic_fadd(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    add_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_fadd.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (add_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = ieee754_add_f32(old_val, add_val & MASK32)
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (add_val & MASK32)
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
    BUFFER_ATOMIC_FADD = "buffer_atomic_fadd"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_fadd instruction."""
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


class Program:
    """Sequence of symbolic instructions."""

    def __init__(self, instructions: list[Instruction] | None = None) -> None:
        self.instructions: list[Instruction] = list(instructions) if instructions else []

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_fadd."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
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
        """Executes a single buffer_atomic_fadd instruction."""
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        ret_val, in_bounds = ref_buffer_atomic_fadd(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            add_val=vdata,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u32(RegKind.VGPR, inst.vdata.index, ret_val)

        self.trace.append(f"{inst.to_assembly()} => ret={ret_val:#010x}, in_bounds={in_bounds}")
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        return [self.step(inst) for inst in prog.instructions]


class TestBufferAtomicFaddOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_float_add_in_bounds(self) -> None:
        # Pre-seed 1.5 (0x3FC00000) at address 100, add 2.5 (0x40200000) -> 4.0 (0x40800000)
        c_1_5 = f32_to_u32(1.5)
        c_2_5 = f32_to_u32(2.5)
        c_4_0 = f32_to_u32(4.0)

        self.mem.write_u32(100, c_1_5)
        ret, in_bounds = ref_buffer_atomic_fadd(
            self.mem, self.desc, soffset=0, vaddr=100, add_val=c_2_5, glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(100), c_4_0)

    def test_atomic_fadd_glc_returns_pre_op_value(self) -> None:
        c_10 = f32_to_u32(10.0)
        c_5 = f32_to_u32(5.0)
        c_15 = f32_to_u32(15.0)

        self.mem.write_u32(200, c_10)
        ret, in_bounds = ref_buffer_atomic_fadd(
            self.mem, self.desc, soffset=0, vaddr=200, add_val=c_5, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, c_10)
        self.assertEqual(self.mem.read_u32(200), c_15)

    def test_out_of_bounds_rejection(self) -> None:
        c_val = f32_to_u32(999.0)
        self.mem.write_u32(1024, c_val)
        ret, in_bounds = ref_buffer_atomic_fadd(
            self.mem, self.desc, soffset=0, vaddr=1024, add_val=f32_to_u32(1.0), glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), c_val)

    def test_signed_zero_addition(self) -> None:
        # +0.0 + +0.0 = +0.0
        self.mem.write_u32(0, POS_ZERO)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, self.desc, soffset=0, vaddr=0, add_val=POS_ZERO, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(0), POS_ZERO)

        # -0.0 + -0.0 = -0.0
        self.mem.write_u32(4, NEG_ZERO)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, self.desc, soffset=0, vaddr=4, add_val=NEG_ZERO, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(4), NEG_ZERO)

        # -0.0 + +0.0 = +0.0
        self.mem.write_u32(8, NEG_ZERO)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, self.desc, soffset=0, vaddr=8, add_val=POS_ZERO, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(8), POS_ZERO)

    def test_infinity_and_nan_arithmetic(self) -> None:
        # (+Inf) + (+Inf) = +Inf
        self.mem.write_u32(0, POS_INF)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, self.desc, soffset=0, vaddr=0, add_val=POS_INF, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(0), POS_INF)

        # (+Inf) + (-Inf) = NaN
        self.mem.write_u32(4, POS_INF)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, self.desc, soffset=0, vaddr=4, add_val=NEG_INF, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(4), QNAN_DEFAULT)

        # 1.0 + NaN = NaN
        self.mem.write_u32(8, f32_to_u32(1.0))
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, self.desc, soffset=0, vaddr=8, add_val=QNAN_DEFAULT, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u32(8), QNAN_DEFAULT)

    def test_inst_offset_and_soffset_combination(self) -> None:
        c_1 = f32_to_u32(100.0)
        c_2 = f32_to_u32(50.0)
        self.mem.write_u32(100, c_1)
        ret, in_bounds = ref_buffer_atomic_fadd(
            self.mem, self.desc, soffset=64, vaddr=20, add_val=c_2, inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, c_1)
        self.assertEqual(self.mem.read_u32(100), f32_to_u32(150.0))


class TestBufferAtomicFaddAST(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_fadd v0, v1, s[4:7], s8 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 3, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_fadd v2, v3, s[12:15], 0 offen",
        )

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FADD,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FADD,
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
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        ))
        expected = "buffer_atomic_fadd v0, v1, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


class TestBufferAtomicFaddScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_atomic_fadd_execution(self) -> None:
        val_initial = f32_to_u32(10.5)
        val_add = f32_to_u32(2.5)
        val_expected = f32_to_u32(13.0)

        self.interp.mem.write_u32(64, val_initial)
        self.interp.write_reg_u32(RegKind.VGPR, 0, val_add)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(self.interp.mem.read_u32(64), val_expected)
        # Without glc, vdata in register remains untouched
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), val_add)

    def test_scalar_atomic_fadd_glc_writeback(self) -> None:
        val_initial = f32_to_u32(50.0)
        val_add = f32_to_u32(15.0)
        val_expected = f32_to_u32(65.0)

        self.interp.mem.write_u32(128, val_initial)
        self.interp.write_reg_u32(RegKind.VGPR, 0, val_add)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_initial)
        # vdata updated to previous float value (50.0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), val_initial)
        # Memory holds 65.0
        self.assertEqual(self.interp.mem.read_u32(128), val_expected)

    def test_scalar_oob_suppression(self) -> None:
        val_initial = f32_to_u32(77.0)
        self.interp.mem.write_u32(1024, val_initial)
        self.interp.write_reg_u32(RegKind.VGPR, 0, f32_to_u32(1.0))
        self.interp.write_reg_u32(RegKind.VGPR, 1, 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0)
        self.assertEqual(self.interp.mem.read_u32(1024), val_initial)

    def test_scalar_program_chain(self) -> None:
        val_0 = f32_to_u32(1.0)
        val_1 = f32_to_u32(2.0)
        val_2 = f32_to_u32(4.0)
        val_final = f32_to_u32(7.0)

        self.interp.mem.write_u32(0, val_0)
        self.interp.write_reg_u32(RegKind.VGPR, 0, val_1)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 2, val_2)

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))

        self.interp.run_program(prog)
        self.assertEqual(self.interp.mem.read_u32(0), val_final)
        self.assertEqual(len(self.interp.trace), 2)


if __name__ == "__main__":
    unittest.main()
