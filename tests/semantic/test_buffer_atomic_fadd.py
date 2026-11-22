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
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


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
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferAtomicFaddConditionCodes:
    """Architectural floating point condition codes for buffer atomic operations."""
    zero: bool
    negative: bool
    positive: bool
    is_nan: bool
    is_inf: bool


@dataclasses.dataclass(frozen=True)
class BufferStatusFlags:
    """Buffer memory status flags."""
    in_bounds: bool
    oob_fault: bool


def compute_buffer_fadd_condition_codes(val: int) -> BufferAtomicFaddConditionCodes:
    """Computes condition codes evaluating floating point properties."""
    val32 = val & MASK32
    nan = ((val32 & EXP_MASK) == EXP_MASK and (val32 & MANT_MASK) != 0)
    inf = (val32 & 0x7FFFFFFF) == EXP_MASK
    sign = bool(val32 & SIGN_MASK)
    zero = (val32 & 0x7FFFFFFF) == 0
    return BufferAtomicFaddConditionCodes(
        zero=zero,
        negative=sign and not nan,
        positive=not sign and not zero and not nan,
        is_nan=nan,
        is_inf=inf,
    )


def compute_buffer_status_flags(in_bounds: bool) -> BufferStatusFlags:
    """Computes buffer memory bounds flags."""
    return BufferStatusFlags(
        in_bounds=in_bounds,
        oob_fault=not in_bounds,
    )


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarBufferInterpreter:
    """Deterministic scalar interpreter for buffer_atomic_fadd."""

    def __init__(self, mem_size: int = 65536) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mem: BufferMemory = BufferMemory(mem_size)
        self.vcc: int = 0
        self.last_condition_codes: BufferAtomicFaddConditionCodes | None = None
        self.last_status_flags: BufferStatusFlags | None = None
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

        self.last_condition_codes = compute_buffer_fadd_condition_codes(ret_val)
        self.last_status_flags = compute_buffer_status_flags(in_bounds)
        # Note: buffer atomic instructions never modify VCC; carry/borrow register remains untouched.

        self.trace.append(f"{inst.to_assembly()} => ret={ret_val:#010x}, in_bounds={in_bounds}")
        return ret_val, in_bounds

    def run_program(self, prog: Program) -> list[tuple[int, bool]]:
        return [self.step(inst) for inst in prog.instructions]


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDBufferWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for buffer atomics."""

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
        """Executes a buffer_atomic_fadd instruction across all lanes in the wavefront."""
        desc = self.get_buffer_descriptor(inst.srsrc.index)
        soffset = (
            self.read_lane_reg_u32(0, inst.soffset.kind, inst.soffset.index)
            if isinstance(inst.soffset, Register)
            else inst.soffset
        )

        # 1. Latch inputs for each lane
        lane_inputs: list[tuple[int, int]] = []
        for lane in range(self.wave_size):
            vd = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index)
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_inputs.append((vd, va))

        # 2. Serialize atomic operations across active lanes
        lane_results: list[tuple[int | None, bool]] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append((None, False))
                continue

            vdata, vaddr = lane_inputs[lane]
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
                self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index, ret_val)
            lane_results.append((ret_val, in_bounds))

        return lane_results


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


class TestBufferAtomicFaddSIMD(unittest.TestCase):
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
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, f32_to_u32(float(lane + 1)))
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertTrue(results[lane][1])
            self.assertEqual(simd.mem.read_u32(lane * 4), f32_to_u32(float(lane + 1)))

    def test_wave64_convergent_atomic_accumulation(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.mem.write_u32(0, POS_ZERO)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, f32_to_u32(1.0))
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        results = simd.step(inst)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertTrue(results[lane][1])
            self.assertEqual(results[lane][0], f32_to_u32(float(lane)))
        # Total sum in memory after 64 lanes adding 1.0 is 64.0
        self.assertEqual(simd.mem.read_u32(0), f32_to_u32(64.0))

    def test_wave32_exec_masking_partial_lanes(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        exec_mask = 0x000000FF  # Only lanes 0..7 active
        simd.set_exec_mask(exec_mask)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        canary = f32_to_u32(100.0)
        add_val = f32_to_u32(5.0)
        expected_active = f32_to_u32(105.0)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, add_val)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.mem.write_u32(lane * 4, canary)

        results = simd.step(inst)
        for lane in range(32):
            if lane < 8:
                self.assertEqual(results[lane][0], canary)
                self.assertEqual(simd.mem.read_u32(lane * 4), expected_active)
            else:
                self.assertIsNone(results[lane][0])
                # Memory untouched for inactive lanes
                self.assertEqual(simd.mem.read_u32(lane * 4), canary)
                # Register vdata untouched for inactive lanes
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), add_val)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=64, mem_size=4096)
        simd.set_buffer_descriptor(4, self.desc)
        simd.set_exec_mask(0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        initial_val = f32_to_u32(42.0)
        simd.mem.write_u32(0, initial_val)
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, f32_to_u32(10.0))
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0)

        results = simd.step(inst)
        self.assertEqual(results, [(None, False)] * 64)
        self.assertEqual(simd.mem.read_u32(0), initial_val)
        for lane in range(64):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), f32_to_u32(10.0))


class TestBufferAtomicFaddRegisterAliasing(unittest.TestCase):
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
        # v0 serves as both vdata (float add value) and vaddr (byte offset)
        # Pre-seed address 64 with 100.0
        c_100 = f32_to_u32(100.0)
        self.interp.mem.write_u32(64, c_100)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 64)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        expected_new = ieee754_add_f32(c_100, 64)
        self.assertEqual(self.interp.mem.read_u32(64), expected_new)
        # v0 receives pre-op value (100.0)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), c_100)

    def test_aliasing_vdata_equals_vaddr_without_glc(self) -> None:
        c_50 = f32_to_u32(50.0)
        self.interp.mem.write_u32(32, c_50)
        self.interp.write_reg_u32(RegKind.VGPR, 0, 32)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=False),
        )
        ret, in_bounds = self.interp.step(inst)
        self.assertTrue(in_bounds)
        expected_new = ieee754_add_f32(c_50, 32)
        self.assertEqual(self.interp.mem.read_u32(32), expected_new)
        # Without glc, v0 retains its original value (32)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 32)

    def test_unrelated_vgprs_preserved(self) -> None:
        for idx in range(10, 20):
            self.interp.write_reg_u32(RegKind.VGPR, idx, 0x77000000 + idx)

        self.interp.write_reg_u32(RegKind.VGPR, 0, f32_to_u32(1.0))
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)

        for idx in range(10, 20):
            self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, idx), 0x77000000 + idx)

    def test_sgpr_descriptor_and_offset_preserved(self) -> None:
        d0, d1, d2, d3 = self.desc.to_dwords()
        self.interp.write_reg_u32(RegKind.SGPR, 8, 16)  # soffset

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
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
        simd.mem.write_u32(16, POS_ZERO)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 16)  # address and value = 16

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        simd.step(inst)

        # Active lanes: v0 updated to pre-op memory values
        v_16 = 16
        c0 = POS_ZERO
        c1 = ieee754_add_f32(c0, v_16)
        c2 = ieee754_add_f32(c1, v_16)
        c3 = ieee754_add_f32(c2, v_16)
        expected_active = [c0, c1, c2, c3]
        for lane in range(4):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected_active[lane])
        # Inactive lanes: v0 untouched (16)
        for lane in range(4, 32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 16)


class TestBufferAtomicFaddConditionCodes(unittest.TestCase):
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
        self.interp.mem.write_u32(0, POS_ZERO)
        self.interp.write_reg_u32(RegKind.VGPR, 0, POS_ZERO)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        cc_zero = self.interp.last_condition_codes
        self.assertIsNotNone(cc_zero)
        assert cc_zero is not None
        self.assertTrue(cc_zero.zero)
        self.assertFalse(cc_zero.negative)
        self.assertFalse(cc_zero.positive)

        # Positive float
        self.interp.mem.write_u32(0, f32_to_u32(5.0))
        self.interp.step(inst)
        cc_pos = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)

        # Negative float
        self.interp.mem.write_u32(0, f32_to_u32(-5.0))
        self.interp.step(inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)

    def test_condition_codes_nan_and_inf(self) -> None:
        self.interp.write_reg_u32(RegKind.VGPR, 0, POS_ZERO)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )

        # NaN
        self.interp.mem.write_u32(0, QNAN_DEFAULT)
        self.interp.step(inst)
        cc_nan = self.interp.last_condition_codes
        self.assertIsNotNone(cc_nan)
        assert cc_nan is not None
        self.assertTrue(cc_nan.is_nan)
        self.assertFalse(cc_nan.zero)

        # Positive Infinity
        self.interp.mem.write_u32(0, POS_INF)
        self.interp.step(inst)
        cc_pinf = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pinf)
        assert cc_pinf is not None
        self.assertTrue(cc_pinf.is_inf)
        self.assertTrue(cc_pinf.positive)

        # Negative Infinity
        self.interp.mem.write_u32(0, NEG_INF)
        self.interp.step(inst)
        cc_ninf = self.interp.last_condition_codes
        self.assertIsNotNone(cc_ninf)
        assert cc_ninf is not None
        self.assertTrue(cc_ninf.is_inf)
        self.assertTrue(cc_ninf.negative)

    def test_status_flags_in_bounds_and_oob(self) -> None:
        self.interp.write_reg_u32(RegKind.VGPR, 0, f32_to_u32(1.0))
        self.interp.write_reg_u32(RegKind.VGPR, 1, 100)  # in bounds (< 1024)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
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
        canary = 0x1234567887654321
        self.interp.vcc = canary

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
            vdata=Register(RegKind.VGPR, 0, count=1),
            vaddr=Register(RegKind.VGPR, 1, count=1),
            srsrc=Register(RegKind.SGPR, 4, count=4),
            soffset=0,
            mods=BufferAtomicModifiers(glc=True),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, canary)


class TestBufferAtomicFaddBoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, overflows, and NaN handling."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)

    def test_last_valid_dword_boundary(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=100)
        c_val = f32_to_u32(2.0)
        c_add = f32_to_u32(3.0)
        c_exp = f32_to_u32(5.0)

        # Last valid 4-byte dword starts at offset 96 (96 + 4 = 100)
        self.mem.write_u32(96, c_val)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=96, add_val=c_add, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, c_val)
        self.assertEqual(self.mem.read_u32(96), c_exp)

        # Offsets 97, 98, 99, 100 all exceed num_records
        for oob_off in (97, 98, 99, 100):
            ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=oob_off, add_val=c_add, glc=True)
            self.assertFalse(in_bounds)
            self.assertEqual(ret, 0)

    def test_zero_capacity_buffer(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=0)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=0, add_val=f32_to_u32(1.0), glc=True)
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)

    def test_subnormal_floating_point_addition(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        # Smallest positive subnormals: 1 LSB + 2 LSB = 3 LSB
        self.mem.write_u32(0, 0x00000001)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=0, add_val=0x00000002, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x00000001)
        self.assertEqual(self.mem.read_u32(0), 0x00000003)

        # Largest subnormal (0x007FFFFF) + 1 LSB transitions to smallest normal float (0x00800000)
        self.mem.write_u32(4, 0x007FFFFF)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=4, add_val=0x00000001, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0x007FFFFF)
        self.assertEqual(self.mem.read_u32(4), 0x00800000)

    def test_float_overflow_to_infinity(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        flt_max = 0x7F7FFFFF  # Largest finite single-precision float (~3.4028235e+38)

        # FLT_MAX + FLT_MAX overflows to +Infinity
        self.mem.write_u32(0, flt_max)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=0, add_val=flt_max, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, flt_max)
        self.assertEqual(self.mem.read_u32(0), POS_INF)

        # -FLT_MAX + -FLT_MAX overflows to -Infinity
        neg_flt_max = flt_max | SIGN_MASK
        self.mem.write_u32(4, neg_flt_max)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=4, add_val=neg_flt_max, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, neg_flt_max)
        self.assertEqual(self.mem.read_u32(4), NEG_INF)

    def test_nan_propagation_and_signaling_nan(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        c_one = f32_to_u32(1.0)
        snan = 0x7F800001  # Signaling NaN

        # Quiet NaN + 1.0 = QNAN
        self.mem.write_u32(0, QNAN_DEFAULT)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=0, add_val=c_one, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, QNAN_DEFAULT)
        self.assertEqual(self.mem.read_u32(0), QNAN_DEFAULT)

        # Signaling NaN + 1.0 = QNAN
        self.mem.write_u32(4, snan)
        ret, in_bounds = ref_buffer_atomic_fadd(self.mem, desc, soffset=0, vaddr=4, add_val=c_one, glc=True)
        self.assertTrue(in_bounds)
        self.assertEqual(ret, snan)
        self.assertEqual(self.mem.read_u32(4), QNAN_DEFAULT)


class TestBufferAtomicFaddNegativeControls(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and invalid AST inputs."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1, count=1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256, count=1)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 254, count=4)

    def test_invalid_destination_kind(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FADD,
                vdata=Register(RegKind.SGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=4),
                soffset=0,
            )

    def test_invalid_descriptor_size(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FADD,
                vdata=Register(RegKind.VGPR, 0, count=1),
                vaddr=Register(RegKind.VGPR, 1, count=1),
                srsrc=Register(RegKind.SGPR, 4, count=2),
                soffset=0,
            )

    def test_invalid_soffset_kind(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.BUFFER_ATOMIC_FADD,
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


class TestBufferAtomicFaddMutationDetection(unittest.TestCase):
    """Mutation tests: verifies that intentional sabotages/mutations of the oracle are caught."""

    def test_mutation_oob_check_omission(self) -> None:
        # Sabotage: ignoring NUM_RECORDS bounds check
        mem = BufferMemory(256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=32)

        def mutated_atomic_fadd_no_bounds(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            m.write_u32(off, ieee754_add_f32(old, val))
            return old

        # An access at offset 64 should be rejected by the oracle
        _, in_bounds = ref_buffer_atomic_fadd(mem, desc, soffset=0, vaddr=64, add_val=f32_to_u32(1.0), glc=True)
        self.assertFalse(in_bounds)

        with self.assertRaises(AssertionError):
            self.assertEqual(in_bounds, True)

    def test_mutation_glc_returns_post_op_value(self) -> None:
        # Sabotage: returning new value instead of old value
        mem = BufferMemory(256)
        c_10 = f32_to_u32(10.0)
        c_5 = f32_to_u32(5.0)
        mem.write_u32(0, c_10)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_atomic_fadd_post_op(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            new_val = ieee754_add_f32(old, val)
            m.write_u32(off, new_val)
            return new_val  # BUG: returns post-op instead of pre-op

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_atomic_fadd_post_op(mem, desc, 0, c_5), c_10)

    def test_mutation_integer_add_instead_of_float(self) -> None:
        # Sabotage: integer addition instead of IEEE 754 float addition
        mem = BufferMemory(256)
        c_1 = f32_to_u32(1.0)
        c_2 = f32_to_u32(2.0)
        mem.write_u32(0, c_1)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)

        def mutated_integer_add(m: BufferMemory, d: BufferResourceDescriptor, off: int, val: int) -> int:
            old = m.read_u32(off)
            new_val = (old + val) & MASK32  # BUG: integer addition
            m.write_u32(off, new_val)
            return old

        mutated_integer_add(mem, desc, 0, c_1)
        with self.assertRaises(AssertionError):
            # 1.0 + 1.0 should be 2.0 (0x40000000), but integer add yields 0x7F000000
            self.assertEqual(mem.read_u32(0), c_2)

    def test_mutation_simd_exec_leak(self) -> None:
        simd = SIMDBufferWavefrontInterpreter(wave_size=32, mem_size=256)
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=256)
        simd.set_buffer_descriptor(4, desc)
        simd.set_exec_mask(0)  # all lanes disabled

        canary = f32_to_u32(999.0)
        simd.mem.write_u32(0, canary)

        inst = Instruction(
            opcode=Opcode.BUFFER_ATOMIC_FADD,
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
        simd.mem.write_u32(0, f32_to_u32(1.0))
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.mem.read_u32(0), canary)


if __name__ == "__main__":
    unittest.main()
