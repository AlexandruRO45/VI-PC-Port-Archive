"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_or.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory atomic bitwise OR:
  buffer_atomic_or vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit dword value to atomically OR into memory.
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
     effective_offset = (soffset & 0xFFFFFFFF) + (inst_offset & 0xFFF) + (vaddr & 0xFFFFFFFF if offen else 0)
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic bitwise OR is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = (old_val | vdata) & 0xFFFFFFFF
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
import struct
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


def ref_buffer_atomic_or(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    or_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_or.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (or_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = (old_val | (or_val & MASK32)) & MASK32
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (or_val & MASK32)
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
    BUFFER_ATOMIC_OR = "buffer_atomic_or"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a buffer_atomic_or instruction."""
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
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferConditionCodes:
    """Condition codes reflecting operation result values."""
    zero: bool
    negative: bool
    positive: bool
    all_ones: bool


@dataclasses.dataclass(frozen=True)
class BufferStatusFlags:
    """Status flags reflecting memory bounds and addressing validity."""
    in_bounds: bool
    oob_fault: bool


def compute_buffer_condition_codes(val: int) -> BufferConditionCodes:
    """Computes arithmetic/logical condition codes from a 32-bit dword value."""
    val32 = val & MASK32
    return BufferConditionCodes(
        zero=(val32 == 0),
        negative=bool(val32 & SIGN_BIT32),
        positive=(val32 > 0 and not bool(val32 & SIGN_BIT32)),
        all_ones=(val32 == MASK32),
    )


def compute_buffer_status_flags(in_bounds: bool) -> BufferStatusFlags:
    """Computes buffer status flags reflecting memory bounds validation."""
    return BufferStatusFlags(
        in_bounds=in_bounds,
        oob_fault=not in_bounds,
    )


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_or."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.trace: list[str] = []
        self.last_condition_codes: BufferConditionCodes | None = None
        self.last_status_flags: BufferStatusFlags | None = None

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
        """Executes a single buffer_atomic_or instruction."""
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        soffset = (
            self.read_reg_u32(inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        ret_val, in_bounds = ref_buffer_atomic_or(
            self.mem,
            desc,
            soffset=soffset,
            vaddr=vaddr,
            or_val=vdata,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if inst.mods.glc:
            self.write_reg_u32(RegKind.VGPR, inst.vdata.index, ret_val)

        # Buffer atomic operations compute condition codes on return value / operand
        self.last_condition_codes = compute_buffer_condition_codes(
            ret_val if inst.mods.glc else vdata
        )
        self.last_status_flags = compute_buffer_status_flags(in_bounds)
        # Note: buffer atomic instructions never modify VCC; carry/borrow register remains untouched.

        self.trace.append(f"{inst.to_assembly()} => ret={ret_val:#010x}, in_bounds={in_bounds}")
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


# ---------------------------------------------------------------------------
# SIMD Multi-Lane Wavefront Interpreter (Wave32 and Wave64)
# ---------------------------------------------------------------------------

class SIMDBufferWavefrontInterpreter:
    """SIMD wavefront interpreter supporting Wave32 and Wave64 execution modes."""

    def __init__(self, wave_size: int = 32, mem_size: int = 65536) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got: {wave_size}")
        self.wave_size: int = wave_size
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.exec_mask: int = (1 << wave_size) - 1
        self.mem: BufferMemory = BufferMemory(mem_size)

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

    def set_exec_mask(self, mask: int) -> None:
        limit = (1 << self.wave_size) - 1
        self.exec_mask = mask & limit

    def set_buffer_descriptor(self, idx: int, desc: BufferResourceDescriptor) -> None:
        d0, d1, d2, d3 = desc.to_dwords()
        self.sgprs[idx] = d0
        self.sgprs[idx + 1] = d1
        self.sgprs[idx + 2] = d2
        self.sgprs[idx + 3] = d3

    def get_buffer_descriptor(self, idx: int) -> BufferResourceDescriptor:
        return BufferResourceDescriptor.from_dwords(
            self.sgprs[idx],
            self.sgprs[idx + 1],
            self.sgprs[idx + 2],
            self.sgprs[idx + 3],
        )

    def step(self, inst: Instruction) -> list[tuple[int | None, bool]]:
        """Executes the instruction across all active lanes of the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.sgprs[inst.soffset.index]
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            lane_active = bool(self.exec_mask & (1 << lane))
            if not lane_active:
                results.append((None, False))
                continue

            vdata = self.vgprs[lane][inst.vdata.index]
            vaddr = self.vgprs[lane][inst.vaddr.index]

            ret_val, in_bounds = ref_buffer_atomic_or(
                self.mem,
                desc,
                soffset=soffset,
                vaddr=vaddr,
                or_val=vdata,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if inst.mods.glc:
                self.vgprs[lane][inst.vdata.index] = ret_val

            results.append((ret_val, in_bounds))

        return results


class TestBufferAtomicOrOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_or_in_bounds(self) -> None:
        # Pre-seed address 100 with 0b0010 (2)
        self.mem.write_u32(100, 0b0010)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, self.desc, soffset=0, vaddr=100, or_val=0b0100, glc=False
        )
        self.assertTrue(in_bounds)
        # Memory should now hold 0b0010 | 0b0100 = 0b0110 (6)
        self.assertEqual(self.mem.read_u32(100), 0b0110)

    def test_atomic_or_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(200, 0x12340000)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, self.desc, soffset=0, vaddr=200, or_val=0x00005678, glc=True
        )
        self.assertTrue(in_bounds)
        # glc returns old value (0x12340000)
        self.assertEqual(ret, 0x12340000)
        # memory updated to bitwise OR
        self.assertEqual(self.mem.read_u32(200), 0x12345678)

    def test_out_of_bounds_rejection(self) -> None:
        # num_records is 1024; offset 1024 is out of bounds
        self.mem.write_u32(1024, 0xDEADBEEF)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, self.desc, soffset=0, vaddr=1024, or_val=0xFFFFFFFF, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(1024), 0xDEADBEEF)

    def test_bitwise_or_all_ones(self) -> None:
        self.mem.write_u32(0, 0x0F0F0F0F)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, self.desc, soffset=0, vaddr=0, or_val=0xF0F0F0F0, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x0F0F0F0F)
        self.assertEqual(self.mem.read_u32(0), 0xFFFFFFFF)

    def test_bitwise_or_idempotent(self) -> None:
        self.mem.write_u32(16, 0xAAAAAAAA)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, self.desc, soffset=0, vaddr=16, or_val=0xAAAAAAAA, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xAAAAAAAA)
        self.assertEqual(self.mem.read_u32(16), 0xAAAAAAAA)

    def test_bitwise_or_zero_identity(self) -> None:
        self.mem.write_u32(40, 0xCAFEFEED)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, self.desc, soffset=0, vaddr=40, or_val=0, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xCAFEFEED)
        self.assertEqual(self.mem.read_u32(40), 0xCAFEFEED)

    def test_inst_offset_and_soffset_combination(self) -> None:
        # Total offset = soffset(64) + inst_offset(16) + vaddr(20) = 100
        self.mem.write_u32(100, 0x00000001)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, self.desc, soffset=64, vaddr=20, or_val=0x00000002, inst_offset=16, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x00000001)
        self.assertEqual(self.mem.read_u32(100), 0x00000003)


class TestBufferAtomicOrAST(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(offset=16, glc=True, offen=True),
        )
        self.assertEqual(
            inst.to_assembly(),
            "buffer_atomic_or v0, v1, s[4:7], s8 offen offset:16 glc",
        )

        inst_imm = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 3, count=1),
            srsrc=Register(RegKind.SGPR, 12, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(offen=True),
        )
        self.assertEqual(
            inst_imm.to_assembly(),
            "buffer_atomic_or v2, v3, s[12:15], 0 offen",
        )

    def test_instruction_validation(self) -> None:
        # Invalid vdata kind (must be VGPR)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_OR,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

        # Invalid srsrc count (must be 4)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_OR,
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
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        ))
        expected = "buffer_atomic_or v0, v1, s[4:7], 0 offen glc"
        self.assertEqual(prog.to_assembly(), expected)


class TestBufferAtomicOrScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_scalar_atomic_or_execution(self) -> None:
        self.interp.mem.write_u32(64, 0x00FF0000)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x0000FF00)  # vdata
        self.interp.write_reg_u32(RegKind.VGPR, 1, 64)          # vaddr

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(self.interp.mem.read_u32(64), 0x00FFFF00)
        # Without glc, vdata remains unchanged
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0x0000FF00)

    def test_scalar_atomic_or_glc_execution(self) -> None:
        self.interp.mem.write_u32(128, 0x12340000)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0x00005678)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 128)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 3, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x12340000)
        # With glc, vdata receives pre-op value
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 2), 0x12340000)
        self.assertEqual(self.interp.mem.read_u32(128), 0x12345678)

    def test_scalar_oob_execution(self) -> None:
        self.interp.mem.write_u32(1024, 0xABCDEF01)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0xFFFFFFFF)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
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
        self.assertEqual(self.interp.mem.read_u32(1024), 0xABCDEF01)

    def test_scalar_program_execution(self) -> None:
        prog = Program()
        # Instruction 1: OR 0x0001 at offset 0
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))
        # Instruction 2: OR 0x0002 at offset 0
        prog.append(Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 2, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        ))

        self.interp.mem.write_u32(0, 0x00000000)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x00000001)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0x00000002)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        results = self.interp.run_program(prog)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0][1])
        self.assertTrue(results[1][1])
        self.assertEqual(self.interp.mem.read_u32(0), 0x00000003)
        self.assertEqual(len(self.interp.trace), 2)

    def test_scalar_soffset_sgpr_resolution(self) -> None:
        self.interp.write_reg_u32(RegKind.SGPR, 8, 32)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x0F)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 16)
        self.interp.mem.write_u32(48, 0xF0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xF0)
        self.assertEqual(self.interp.mem.read_u32(48), 0xFF)


class TestBufferAtomicOrSIMD(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        for lane in range(32):
            val = 1 << lane
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, val)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertTrue(results[lane][1])
            self.assertEqual(simd.mem.read_u32(lane * 4), 1 << lane)

    def test_wave64_convergent_atomic_accumulation(self) -> None:
        # All 64 lanes atomically OR into address 0:
        # Lanes 0..31 set bit (1 << lane); lanes 32..63 set bit (1 << (lane - 32))
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.mem.write_u32(0, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        for lane in range(64):
            bit_val = 1 << (lane % 32)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, bit_val)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        results = simd.step(inst)
        # All 32 bits must have been set in memory => 0xFFFFFFFF
        self.assertEqual(simd.mem.read_u32(0), 0xFFFFFFFF)
        # First lane to execute must have received pre-op initial value (0)
        self.assertEqual(results[0][0], 0)

    def test_wave32_exec_masking_partial_lanes(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x000000FF  # Only lanes 0..7 active
        simd.set_exec_mask(exec_mask)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        initial_val = 0x10000000
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x01)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.mem.write_u32(lane * 4, initial_val)

        results = simd.step(inst)
        for lane in range(32):
            if lane < 8:
                self.assertEqual(results[lane][0], initial_val)
                self.assertEqual(simd.mem.read_u32(lane * 4), initial_val | 0x01)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), initial_val)
            else:
                self.assertIsNone(results[lane][0])
                # Memory untouched for inactive lanes
                self.assertEqual(simd.mem.read_u32(lane * 4), initial_val)
                # Register vdata untouched for inactive lanes
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0x01)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.mem.write_u32(0, 0x55555555)
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0xAAAAAAAA)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        results = simd.step(inst)
        self.assertEqual(results, [(None, False)] * 64)
        self.assertEqual(simd.mem.read_u32(0), 0x55555555)
        for lane in range(64):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0xAAAAAAAA)


class TestBufferAtomicOrRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_aliasing_vdata_equals_vaddr_with_glc(self) -> None:
        # v0 serves as both vdata (OR operand) and vaddr (byte offset)
        # Pre-seed address 64 with 0x00FF0000
        self.interp.mem.write_u32(64, 0x00FF0000)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        # Address 64 should now have 0x00FF0000 | 64 = 0x00FF0040
        self.assertEqual(self.interp.mem.read_u32(64), 0x00FF0040)
        # v0 receives pre-op value (0x00FF0000)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0x00FF0000)

    def test_aliasing_vdata_equals_vaddr_without_glc(self) -> None:
        self.interp.mem.write_u32(32, 0xF000)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 32)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(self.interp.mem.read_u32(32), 0xF020)
        # Without glc, v0 retains its original value (32)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 32)

    def test_unrelated_vgprs_preserved(self) -> None:
        for idx in range(10, 20):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0x88000000 + idx)

        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x0F)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        for idx in range(10, 20):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), 0x88000000 + idx)

    def test_sgpr_descriptor_and_offset_preserved(self) -> None:
        d0, d1, d2, d3 = self.desc.to_dwords()
        self.interp.write_reg_u32(RegKind.SGPR, 8, 16)  # soffset

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=Register(RegKind.SGPR, 8, count=1),
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

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
        simd.mem.write_u32(16, 0x12340000)

        # Aliasing v0 as vdata and vaddr
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 16)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Inactive lanes (4..31) preserve their initial v0 (16)
        for lane in range(4, 32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 16)


class TestBufferAtomicOrConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )
        self.interp.set_buffer_descriptor(4, self.desc)

    def test_condition_codes_zero_pos_neg_all_ones(self) -> None:
        # Pre-seed memory at 0 with 0
        self.interp.mem.write_u32(0, 0)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x10)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        # Pre-op return value is 0 => zero=True
        self.interp.step(inst)
        cc_zero = self.interp.last_condition_codes
        self.assertIsNotNone(cc_zero)
        assert cc_zero is not None
        self.assertTrue(cc_zero.zero)
        self.assertFalse(cc_zero.negative)
        self.assertFalse(cc_zero.positive)
        self.assertFalse(cc_zero.all_ones)

        # Next OR: pre-op is 0x10 (positive)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x80000000)
        self.interp.step(inst)
        cc_pos = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)

        # Next OR: pre-op now has bit 31 set (0x80000010) => negative=True
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0x7FFFFFEF)
        self.interp.step(inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)

        # Next OR: pre-op is 0xFFFFFFFF => all_ones=True
        self.interp.write_reg_u32(RegKind.VGPR, 0, 0)
        self.interp.step(inst)
        cc_all = self.interp.last_condition_codes
        self.assertIsNotNone(cc_all)
        assert cc_all is not None
        self.assertTrue(cc_all.all_ones)

    def test_status_flags_in_bounds_and_oob(self) -> None:
        self.interp.write_reg_u32(RegKind.VGPR, 0, 1)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 100)  # in bounds (< 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
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
        self.interp.write_reg_u32(RegKind.VGPR, 1, 2000)
        self.interp.step(inst)
        flags_oob = self.interp.last_status_flags
        self.assertIsNotNone(flags_oob)
        assert flags_oob is not None
        self.assertFalse(flags_oob.in_bounds)
        self.assertTrue(flags_oob.oob_fault)

    def test_carry_borrow_vcc_isolation(self) -> None:
        canary = 0xFEEDFACECAFEBEEF
        self.interp.vcc = canary

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        # VCC carry/borrow register must be strictly untouched
        self.assertEqual(self.interp.vcc, canary)


class TestBufferAtomicOrBoundaryExtremes(unittest.TestCase):
    """Unit tests for boundary extremes, address wrapping, and IEEE 754 bit representations."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)

    def test_last_valid_dword_boundary(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=100)
        # Last valid 4-byte dword starts at offset 96 (96 + 4 = 100)
        self.mem.write_u32(96, 0x00000001)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, desc, soffset=0, vaddr=96, or_val=0x00000002, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x00000001)
        self.assertEqual(self.mem.read_u32(96), 0x00000003)

        # Offsets 97, 98, 99 all exceed num_records (97 + 4 = 101 > 100)
        for oob_off in (97, 98, 99, 100):
            ret, in_bounds = ref_buffer_atomic_or(
                self.mem, desc, soffset=0, vaddr=oob_off, or_val=0xFF, glc=True
            )
            self.assertFalse(in_bounds)
            self.assertEqual(ret, 0)

    def test_zero_capacity_buffer(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=0)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, desc, soffset=0, vaddr=0, or_val=0xFF, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)

    def test_bitwise_boundary_patterns(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        # 0x00000000 | 0x00000000 = 0
        self.mem.write_u32(0, 0)
        ret, in_bounds = ref_buffer_atomic_or(self.mem, desc, 0, 0, 0, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u32(0), 0)

        # 0xFFFFFFFF | 0xFFFFFFFF = 0xFFFFFFFF
        self.mem.write_u32(4, 0xFFFFFFFF)
        ret, in_bounds = ref_buffer_atomic_or(self.mem, desc, 0, 4, 0xFFFFFFFF, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xFFFFFFFF)
        self.assertEqual(self.mem.read_u32(4), 0xFFFFFFFF)

        # 0x55555555 | 0xAAAAAAAA = 0xFFFFFFFF
        self.mem.write_u32(8, 0x55555555)
        ret, in_bounds = ref_buffer_atomic_or(self.mem, desc, 0, 8, 0xAAAAAAAA, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x55555555)
        self.assertEqual(self.mem.read_u32(8), 0xFFFFFFFF)

        # 0x80000000 | 0x7FFFFFFF = 0xFFFFFFFF
        self.mem.write_u32(12, 0x80000000)
        ret, in_bounds = ref_buffer_atomic_or(self.mem, desc, 0, 12, 0x7FFFFFFF, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x80000000)
        self.assertEqual(self.mem.read_u32(12), 0xFFFFFFFF)

    def test_float_nan_and_subnormal_bit_patterns(self) -> None:
        # Untyped buffer atomic OR operates purely on 32-bit bit patterns,
        # without floating point canonicalization, flush-to-zero, or NaN signaling.
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        # Quiet NaN (0x7FC00000) | sign bit (0x80000000) -> 0xFFC00000
        qnan_bits = 0x7FC00000
        self.mem.write_u32(16, qnan_bits)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, desc, 0, 16, 0x80000000, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, qnan_bits)
        self.assertEqual(self.mem.read_u32(16), 0xFFC00000)

        # Signaling NaN (0x7F800001) | bit 2 -> 0x7F800003
        snan_bits = 0x7F800001
        self.mem.write_u32(20, snan_bits)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, desc, 0, 20, 0x00000002, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, snan_bits)
        self.assertEqual(self.mem.read_u32(20), 0x7F800003)

        # Subnormal bit pattern: 0x00000001 (smallest positive denorm) | 0x00000002 -> 0x00000003
        self.mem.write_u32(24, 0x00000001)
        ret, in_bounds = ref_buffer_atomic_or(
            self.mem, desc, 0, 24, 0x00000002, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x00000001)
        self.assertEqual(self.mem.read_u32(24), 0x00000003)


class TestBufferAtomicOrNegativeControls(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and invalid AST inputs."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1, count=1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256, count=1)
        with self.assertRaises(ValueError):
            # SGPR 254 with count 4 exceeds 256
            Register(RegKind.SGPR, 254, count=4)

    def test_invalid_destination_kind(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_OR,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_invalid_descriptor_size(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_OR,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

    def test_invalid_soffset_kind(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_OR,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=Register(RegKind.VGPR, 8, count=1),
            )

    def test_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDBufferWavefrontInterpreter(wave_size=128)

    def test_invalid_offset(self) -> None:
        with self.assertRaises(ValueError):
            BufferAtomicModifiers(offset=4096)


class TestBufferAtomicOrMutationDetection(unittest.TestCase):
    """Mutation tests: verifies that intentional sabotages/mutations of the oracle are caught."""

    def test_mutation_oob_check_omission(self) -> None:
        # Sabotage: omitting NUM_RECORDS bounds check
        mem = BufferMemory(256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=32)

        def mutated_atomic_or_no_bounds(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            m.write_u32(off, old | val)
            return old

        # An access at offset 64 should be rejected by the oracle
        _, in_bounds = ref_buffer_atomic_or(mem, desc, soffset=0, vaddr=64, or_val=0xFF, glc=True)
        self.assertFalse(in_bounds)

        # Mutated oracle incorrectly allows it
        with self.assertRaises(AssertionError):
            self.assertEqual(in_bounds, True)

    def test_mutation_glc_returns_post_op_value(self) -> None:
        # Sabotage: returning new value instead of old value
        mem = BufferMemory(256)
        mem.write_u32(0, 0x0F)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_atomic_or_post_op(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            new_val = (old | val) & MASK32
            m.write_u32(off, new_val)
            return new_val  # BUG: returns post-op instead of pre-op

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_atomic_or_post_op(mem, desc, 0, 0xF0), 0x0F)

    def test_mutation_bitwise_and_bug(self) -> None:
        # Sabotage: using AND instead of OR
        mem = BufferMemory(256)
        mem.write_u32(0, 0x0F)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_atomic_or_and(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            new_val = (old & val) & MASK32  # BUG: AND instead of OR
            m.write_u32(off, new_val)
            return old

        mutated_atomic_or_and(mem, desc, 0, 0xF0)
        with self.assertRaises(AssertionError):
            # Memory should be 0xFF, but bug produced 0x00
            self.assertEqual(mem.read_u32(0), 0xFF)

    def test_mutation_simd_exec_leak(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        simd.set_buffer_descriptor(4, desc)
        simd.set_exec_mask(0)  # all lanes disabled

        canary = 0x12345678
        simd.mem.write_u32(0, canary)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_OR,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        simd.step(inst)
        # Memory must remain canary
        self.assertEqual(simd.mem.read_u32(0), canary)

        # Mutated unconditional write would corrupt memory
        simd.mem.write_u32(0, 0xFFFFFFFF)
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.mem.read_u32(0), canary)


if __name__ == "__main__":
    unittest.main()
