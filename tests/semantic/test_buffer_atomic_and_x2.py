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


# ---------------------------------------------------------------------------
# Condition Codes and Status Flags
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferConditionCodes:
    """Condition codes reflecting 64-bit bitwise operation result values."""
    zero: bool
    negative: bool
    positive: bool


@dataclasses.dataclass(frozen=True)
class BufferStatusFlags:
    """Status flags reflecting memory bounds and addressing validity."""
    in_bounds: bool
    oob_fault: bool


def compute_buffer_condition_codes(pre_op: int, and_val: int, result: int) -> BufferConditionCodes:
    """Computes condition codes from a 64-bit bitwise operation."""
    val64 = result & MASK64
    return BufferConditionCodes(
        zero=(val64 == 0),
        negative=bool(val64 & SIGN_BIT64),
        positive=(val64 > 0 and not bool(val64 & SIGN_BIT64)),
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
    """Deterministic scalar interpreter for buffer_atomic_and_x2."""

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
        if kind == RegKind.VGPR:
            return self.vgprs[idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_reg_u64(self, kind: RegKind, idx: int) -> int:
        lo = self.read_reg_u32(kind, idx)
        hi = self.read_reg_u32(kind, idx + 1)
        return (hi << 32) | lo

    def write_reg_u64(self, kind: RegKind, idx: int, val: int) -> None:
        self.write_reg_u32(kind, idx, val & MASK32)
        self.write_reg_u32(kind, idx + 1, (val >> 32) & MASK32)

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
        """Executes a single buffer_atomic_and_x2 instruction."""
        vdata = self.read_reg_u64(RegKind.VGPR, inst.vdata.index)
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index) if inst.mods.offen else 0
        soffset = (
            self.read_reg_u32(RegKind.SGPR, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )
        desc = self.get_buffer_descriptor(inst.srsrc.index)

        ret_val, in_bounds = ref_buffer_atomic_and_x2(
            mem=self.mem,
            desc=desc,
            soffset=soffset,
            vaddr=vaddr,
            and_val=vdata,
            inst_offset=inst.mods.offset,
            glc=inst.mods.glc,
            offen=inst.mods.offen,
        )

        if in_bounds and inst.mods.glc:
            self.write_reg_u64(RegKind.VGPR, inst.vdata.index, ret_val)

        self.last_condition_codes = compute_buffer_condition_codes(
            ret_val if inst.mods.glc else 0, vdata, ret_val
        )
        self.last_status_flags = compute_buffer_status_flags(in_bounds)

        self.trace.append(f"{inst.to_assembly()} => ret={ret_val:#018x}, in_bounds={in_bounds}")
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestBufferAtomicAndX2ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarBufferInterpreter(mem_size=4096)
        self.desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=4096)
        self.interp.set_buffer_descriptor(0, self.desc)

    def test_scalar_step_in_bounds_glc_true(self) -> None:
        self.interp.mem.write_u64(32, 0x1234567890ABCDEF)
        self.interp.write_reg_u64(RegKind.VGPR, 0, 0x00000000FFFFFFFF)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 32)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x1234567890ABCDEF)
        self.assertEqual(self.interp.mem.read_u64(32), 0x0000000090ABCDEF)
        self.assertEqual(self.interp.read_reg_u64(RegKind.VGPR, 0), 0x1234567890ABCDEF)

    def test_scalar_step_in_bounds_glc_false(self) -> None:
        self.interp.mem.write_u64(64, 0xFFFFFFFFFFFFFFFF)
        self.interp.write_reg_u64(RegKind.VGPR, 4, 0xAAAAAAAA55555555)
        self.interp.write_reg_u32(RegKind.VGPR, 6, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 4, count=2),
            vaddr=Register(RegKind.VGPR, 6),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.interp.mem.read_u64(64), 0xAAAAAAAA55555555)
        self.assertEqual(self.interp.read_reg_u64(RegKind.VGPR, 4), 0xAAAAAAAA55555555)

    def test_scalar_step_oob(self) -> None:
        self.interp.write_reg_u64(RegKind.VGPR, 0, 0xFF)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 5000)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.interp.read_reg_u64(RegKind.VGPR, 0), 0xFF)

    def test_scalar_with_register_soffset(self) -> None:
        self.interp.mem.write_u64(80, 0xFF00FF00FF00FF00)
        self.interp.write_reg_u64(RegKind.VGPR, 0, 0x0F0F0F0F0F0F0F0F)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 40)
        self.interp.write_reg_u32(RegKind.SGPR, 4, 40)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=Register(RegKind.SGPR, 4),
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xFF00FF00FF00FF00)
        self.assertEqual(self.interp.mem.read_u64(80), 0x0F000F000F000F00)


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

    def read_lane_reg_u64(self, lane: int, reg_idx: int) -> int:
        lo = self.vgprs[lane][reg_idx] & MASK32
        hi = self.vgprs[lane][reg_idx + 1] & MASK32
        return (hi << 32) | lo

    def write_lane_reg_u64(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = val & MASK32
        self.vgprs[lane][reg_idx + 1] = (val >> 32) & MASK32

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
        d0 = self.sgprs[idx]
        d1 = self.sgprs[idx + 1]
        d2 = self.sgprs[idx + 2]
        d3 = self.sgprs[idx + 3]
        return BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)

    def step(self, inst: Instruction) -> list[tuple[int, bool]]:
        """Executes a buffer_atomic_and_x2 instruction across all active lanes."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.sgprs[inst.soffset.index]
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )
        results: list[tuple[int, bool]] = []

        for lane in range(self.wave_size):
            lane_active = bool((self.exec_mask >> lane) & 1)
            if not lane_active:
                results.append((0, False))
                continue

            vdata = self.read_lane_reg_u64(lane, inst.vdata.index)
            vaddr = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index) if inst.mods.offen else 0

            ret_val, in_bounds = ref_buffer_atomic_and_x2(
                mem=self.mem,
                desc=desc,
                soffset=soffset,
                vaddr=vaddr,
                and_val=vdata,
                inst_offset=inst.mods.offset,
                glc=inst.mods.glc,
                offen=inst.mods.offen,
            )

            if in_bounds and inst.mods.glc:
                self.write_lane_reg_u64(lane, inst.vdata.index, ret_val)

            results.append((ret_val, in_bounds))

        return results


class TestBufferAtomicAndX2SIMD(unittest.TestCase):
    """Unit tests for SIMD execution divergence across Wave32 and Wave64 wavefronts."""

    def setUp(self) -> None:
        self.desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=8192)

    def test_simd_wave32_all_active(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=8192)
        simd.set_buffer_descriptor(0, self.desc)

        for lane in range(32):
            addr = lane * 8
            simd.mem.write_u64(addr, 0xFFFFFFFFFFFFFFFF)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)
            simd.write_lane_reg_u64(lane, 0, 1 << lane)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            ret, in_bounds = results[lane]
            self.assertTrue(in_bounds)
            self.assertEqual(ret, 0xFFFFFFFFFFFFFFFF)
            self.assertEqual(simd.mem.read_u64(lane * 8), 1 << lane)
            self.assertEqual(simd.read_lane_reg_u64(lane, 0), 0xFFFFFFFFFFFFFFFF)

    def test_simd_wave32_divergent_alternating_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=8192)
        simd.set_buffer_descriptor(0, self.desc)
        simd.set_exec_mask(0x55555555)

        for lane in range(32):
            addr = lane * 8
            simd.mem.write_u64(addr, 0xFF00FF00FF00FF00)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)
            simd.write_lane_reg_u64(lane, 0, 0x0F0F0F0F0F0F0F0F)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)
        for lane in range(32):
            ret, in_bounds = results[lane]
            if lane % 2 == 0:
                self.assertTrue(in_bounds)
                self.assertEqual(ret, 0xFF00FF00FF00FF00)
                self.assertEqual(simd.mem.read_u64(lane * 8), 0x0F000F000F000F00)
                self.assertEqual(simd.read_lane_reg_u64(lane, 0), 0xFF00FF00FF00FF00)
            else:
                self.assertFalse(in_bounds)
                self.assertEqual(simd.mem.read_u64(lane * 8), 0xFF00FF00FF00FF00)
                self.assertEqual(simd.read_lane_reg_u64(lane, 0), 0x0F0F0F0F0F0F0F0F)

    def test_simd_wave64_divergent_half_mask(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=8192)
        simd.set_buffer_descriptor(0, self.desc)
        simd.set_exec_mask(0x00000000FFFFFFFF)

        for lane in range(64):
            addr = lane * 8
            simd.mem.write_u64(addr, MASK64)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, addr)
            simd.write_lane_reg_u64(lane, 0, 0x12345678)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_AND_X2,
            vdata=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2),
            srsrc=Register(RegKind.SGPR, 0, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        results = simd.step(inst)
        for lane in range(64):
            ret, in_bounds = results[lane]
            if lane < 32:
                self.assertTrue(in_bounds)
                self.assertEqual(ret, MASK64)
                self.assertEqual(simd.mem.read_u64(lane * 8), 0x12345678)
            else:
                self.assertFalse(in_bounds)
                self.assertEqual(simd.mem.read_u64(lane * 8), MASK64)
                self.assertEqual(simd.read_lane_reg_u64(lane, 0), 0x12345678)


if __name__ == "__main__":
    unittest.main()
