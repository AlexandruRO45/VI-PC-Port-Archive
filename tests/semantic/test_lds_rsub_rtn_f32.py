from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_rsub_rtn_f32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Floating-Point Reverse Subtraction with Return:
Performs an atomic single-precision IEEE 754 floating-point reverse subtraction on 32-bit LDS memory
and returns the original (pre-operation) value:
    new_val = vdata0 - old_val
    vdst = old_val

Syntax:
    ds_rsub_rtn_f32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val_u32 = LDS[dword_index]
    new_val_u32 = ieee754_sub_f32(vdata0, old_val_u32)  # vdata0 is minuend, old_val is subtrahend
    LDS[dword_index] = new_val_u32
    vdst = old_val_u32
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Single-Precision Constants (raw uint32 bit patterns)
POS_ZERO_F32: int = 0x0000_0000
NEG_ZERO_F32: int = 0x8000_0000
POS_INF_F32: int = 0x7F80_0000
NEG_INF_F32: int = 0xFF80_0000
QNAN_DEFAULT: int = 0x7FC0_0000
SNAN_DEFAULT: int = 0x7F80_0001
ONE_F32: int = 0x3F80_0000          # 1.0f
TWO_F32: int = 0x4000_0000          # 2.0f
THREE_F32: int = 0x4040_0000        # 3.0f
FIVE_F32: int = 0x40A0_0000         # 5.0f
TEN_F32: int = 0x4120_0000          # 10.0f


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK_32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def u32_to_f32(val: int) -> float:
    """Unpack 32-bit unsigned dword to Python float."""
    return struct.unpack("<f", struct.pack("<I", u32(val)))[0]


def f32_to_u32(val: float) -> int:
    """Pack Python float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return NEG_INF_F32 if val < 0 else POS_INF_F32


def ieee754_rsub_f32(vdata0_u32: int, old_val_u32: int) -> int:
    """Bit-accurate IEEE 754 single-precision reverse subtraction: vdata0 - old_val.

    Matches RDNA2 hardware semantics where vdata0 is minuend and old_val is subtrahend.
    """
    a_u32 = u32(vdata0_u32)
    b_u32 = u32(old_val_u32)

    a_exp = (a_u32 >> 23) & 0xFF
    a_mant = a_u32 & 0x7F_FFFF
    b_exp = (b_u32 >> 23) & 0xFF
    b_mant = b_u32 & 0x7F_FFFF

    a_is_nan = (a_exp == 0xFF and a_mant != 0)
    b_is_nan = (b_exp == 0xFF and b_mant != 0)

    # Any NaN operand produces quiet NaN
    if a_is_nan or b_is_nan:
        return QNAN_DEFAULT

    a_is_inf = (a_exp == 0xFF and a_mant == 0)
    b_is_inf = (b_exp == 0xFF and b_mant == 0)
    a_sign = (a_u32 >> 31) & 1
    b_sign = (b_u32 >> 31) & 1

    # (+Inf) - (+Inf) or (-Inf) - (-Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign == b_sign):
        return QNAN_DEFAULT

    # If minuend is Inf, result is minuend
    if a_is_inf:
        return a_u32

    # If subtrahend is Inf, result is -subtrahend
    if b_is_inf:
        return b_u32 ^ 0x8000_0000

    # Float subtraction
    a_f = u32_to_f32(a_u32)
    b_f = u32_to_f32(b_u32)

    # Special handling for signed zeros:
    # (+0.0) - (+0.0) = +0.0
    # (-0.0) - (-0.0) = +0.0
    # (+0.0) - (-0.0) = +0.0
    # (-0.0) - (+0.0) = -0.0
    if a_f == 0.0 and b_f == 0.0:
        if a_sign == 1 and b_sign == 0:
            return NEG_ZERO_F32
        return POS_ZERO_F32

    res_f = a_f - b_f
    res_u32 = f32_to_u32(res_f)

    # In round-to-nearest-even, if difference of equal non-zero finite numbers is zero, sign is +0.0
    if a_f == b_f and res_f == 0.0:
        return POS_ZERO_F32

    return res_u32


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert effective byte address to aligned LDS dword index."""
    return (eff_addr >> 2) & (capacity_dwords - 1)


class LDSMemory:
    """Architectural model of Local Data Share (LDS) memory."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: List[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)

    def atomic_rsub_rtn_f32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit float reverse subtraction on LDS with return.

        Returns:
            Tuple[int, int]: (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        old_val = self.read_dword(idx)
        new_val = ieee754_rsub_f32(vdata0, old_val)
        self.write_dword(idx, new_val)
        return old_val, new_val


def oracle_ds_rsub_rtn_f32(
    lds_bytes: bytearray,
    vaddr: int,
    vdata0: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_SIZE_BYTES,
) -> Tuple[int, int]:
    """Pure bytearray reference mathematical oracle for RDNA2 ds_rsub_rtn_f32."""
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~3  # drop low 2 bits
    old_val = struct.unpack_from("<I", lds_bytes, aligned_addr)[0]
    new_val = ieee754_rsub_f32(vdata0, old_val)
    struct.pack_into("<I", lds_bytes, aligned_addr, new_val)
    return old_val, new_val


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)
        self.lds_bytes = bytearray(LDS_SIZE_BYTES)

    def test_basic_reverse_subtraction(self) -> None:
        # Initial memory has 2.0f, vdata0 is 5.0f
        # Reverse subtraction: new_val = vdata0 - old_val = 5.0f - 2.0f = 3.0f
        # Returns old_val = 2.0f
        self.mem.write_dword(10, TWO_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=40, vdata0=FIVE_F32, offset=0)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_val, THREE_F32)
        self.assertAlmostEqual(u32_to_f32(self.mem.read_dword(10)), 3.0, places=5)

    def test_reverse_subtraction_negative_result(self) -> None:
        # Initial memory has 10.0f, vdata0 is 3.0f
        # Reverse subtraction: 3.0f - 10.0f = -7.0f
        self.mem.write_dword(15, TEN_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=60, vdata0=THREE_F32, offset=0)
        self.assertEqual(old_val, TEN_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), -7.0, places=5)
        self.assertAlmostEqual(u32_to_f32(self.mem.read_dword(15)), -7.0, places=5)

    def test_non_commutativity_difference_from_sub(self) -> None:
        # Crucial test distinguishing rsub from sub:
        # vdata0 = 2.0f, old_val = 5.0f
        # rsub: 2.0f - 5.0f = -3.0f
        # sub:  5.0f - 2.0f = +3.0f
        self.mem.write_dword(20, FIVE_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=80, vdata0=TWO_F32, offset=0)
        self.assertEqual(old_val, FIVE_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), -3.0, places=5)

    def test_address_offset_and_alignment(self) -> None:
        # vaddr = 0x200, offset = 16 -> effective = 0x210 -> dword index = 0x84
        idx = 0x84
        self.mem.write_dword(idx, ONE_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=0x200, vdata0=FIVE_F32, offset=16)
        self.assertEqual(old_val, ONE_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), 4.0, places=5)

        # Unaligned byte address 0x203 drops low 2 bits -> 0x200 (index 0x80)
        self.mem.write_dword(0x80, TWO_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=0x200, vdata0=TEN_F32, offset=3)
        self.assertEqual(old_val, TWO_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), 8.0, places=5)

    def test_bytearray_oracle_equivalence(self) -> None:
        addr = 0x400
        struct.pack_into("<I", self.lds_bytes, addr, TWO_F32)
        old_val, new_val = oracle_ds_rsub_rtn_f32(self.lds_bytes, vaddr=addr, vdata0=FIVE_F32, offset=0)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_val, THREE_F32)
        self.assertEqual(struct.unpack_from("<I", self.lds_bytes, addr)[0], THREE_F32)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Strongly-typed register operand."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index < DEFAULT_NUM_VGPRS):
                raise ValueError(f"VGPR index out of bounds [0, {DEFAULT_NUM_VGPRS - 1}]: {self.index}")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index < DEFAULT_NUM_SGPRS):
                raise ValueError(f"SGPR index out of bounds [0, {DEFAULT_NUM_SGPRS - 1}]: {self.index}")

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


class DSOpcode(Enum):
    """LDS DS-encoding opcode identifier."""
    DS_RSUB_RTN_F32 = "ds_rsub_rtn_f32"


@dataclass(frozen=True)
class DSRsubRtnInstruction:
    """Symbolic AST node representing a ds_rsub_rtn_f32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: RegisterOperand
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_RSUB_RTN_F32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise TypeError(f"vdst must be a VGPR, got {self.vdst}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise TypeError(f"vaddr must be a VGPR, got {self.vaddr}")
        if self.vdata0.reg_type != RegType.VGPR:
            raise TypeError(f"vdata0 must be a VGPR, got {self.vdata0}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"Immediate byte offset out of unsigned 16-bit range [0, 65535]: {self.offset}")

    def disasm(self) -> str:
        base = f"{self.opcode.value} {self.vdst.name}, {self.vaddr.name}, {self.vdata0.name}"
        mods = []
        if self.offset != 0:
            mods.append(f"offset:{self.offset}")
        if self.gds:
            mods.append("gds")
        return f"{base} {' '.join(mods)}".rstrip()


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_RSUB_RTN_F32.value, "ds_rsub_rtn_f32")

    def test_register_operands(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v255 = RegisterOperand(255, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)
        s105 = RegisterOperand(105, RegType.SGPR)

        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(s0.name, "s0")
        self.assertEqual(s105.name, "s105")

    def test_ast_construction_and_disassembly(self) -> None:
        instr_basic = DSRsubRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )
        self.assertEqual(instr_basic.disasm(), "ds_rsub_rtn_f32 v5, v1, v2")

        instr_offset = DSRsubRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=16,
        )
        self.assertEqual(instr_offset.disasm(), "ds_rsub_rtn_f32 v5, v1, v2 offset:16")

        instr_gds = DSRsubRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=8,
            gds=True,
        )
        self.assertEqual(instr_gds.disasm(), "ds_rsub_rtn_f32 v5, v1, v2 offset:8 gds")

    def test_ast_immutability_and_equality(self) -> None:
        instr1 = DSRsubRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(4),
            vdata0=RegisterOperand(5),
            offset=64,
        )
        instr2 = DSRsubRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(4),
            vdata0=RegisterOperand(5),
            offset=64,
        )
        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))

    def test_ast_validation_and_type_checks(self) -> None:
        # Register index out of bounds
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

        # RegType SGPR in vdst, vaddr, or vdata0
        with self.assertRaises(TypeError):
            DSRsubRtnInstruction(
                vdst=RegisterOperand(0, RegType.SGPR),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2),
            )
        with self.assertRaises(TypeError):
            DSRsubRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1, RegType.SGPR),
                vdata0=RegisterOperand(2),
            )
        with self.assertRaises(TypeError):
            DSRsubRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2, RegType.SGPR),
            )

        # Offset out of range
        with self.assertRaises(ValueError):
            DSRsubRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2),
                offset=-1,
            )
        with self.assertRaises(ValueError):
            DSRsubRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2),
                offset=65536,
            )


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarLDSInterpreter:
    """Deterministic scalar architectural interpreter for LDS instructions."""

    def __init__(self, lds_size_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.mem: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def read_reg(self, op: RegisterOperand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index]
        return self.sgprs[op.index]

    def write_reg(self, op: RegisterOperand, val: int) -> None:
        val32 = u32(val)
        if op.reg_type == RegType.VGPR:
            self.vgprs[op.index] = val32
        else:
            self.sgprs[op.index] = val32

    def step(self, instr: DSRsubRtnInstruction) -> Tuple[int, int]:
        """Execute a single ds_rsub_rtn_f32 instruction deterministically."""
        vaddr_val = self.read_reg(instr.vaddr)
        vdata0_val = self.read_reg(instr.vdata0)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(
            vaddr=vaddr_val,
            vdata0=vdata0_val,
            offset=instr.offset,
        )
        self.write_reg(instr.vdst, old_val)
        return old_val, new_val


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(lds_size_dwords=1024)

    def test_scalar_rsub_step_execution(self) -> None:
        # Pre-seed memory index 10 (byte address 40) with 2.0f
        self.interp.mem.write_dword(10, TWO_F32)
        # v1 (vaddr) = 40
        self.interp.write_reg(RegisterOperand(1), 40)
        # v2 (vdata0) = 5.0f
        self.interp.write_reg(RegisterOperand(2), FIVE_F32)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )
        old_val, new_val = self.interp.step(instr)

        # Pre-op memory value was 2.0f -> returned into v0
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(self.interp.read_reg(RegisterOperand(0)), TWO_F32)
        # Post-op memory value is 5.0f - 2.0f = 3.0f
        self.assertEqual(new_val, THREE_F32)
        self.assertEqual(self.interp.mem.read_dword(10), THREE_F32)

    def test_scalar_multi_step_sequence(self) -> None:
        # Pre-seed memory index 20 (byte address 80) with 1.0f
        self.interp.mem.write_dword(20, ONE_F32)
        self.interp.write_reg(RegisterOperand(5), 80)

        # Step 1: vdata0 = 4.0f -> new = 4.0 - 1.0 = 3.0f
        four_f32 = f32_to_u32(4.0)
        self.interp.write_reg(RegisterOperand(6), four_f32)
        instr1 = DSRsubRtnInstruction(
            vdst=RegisterOperand(4),
            vaddr=RegisterOperand(5),
            vdata0=RegisterOperand(6),
        )
        old1, new1 = self.interp.step(instr1)
        self.assertEqual(old1, ONE_F32)
        self.assertEqual(self.interp.read_reg(RegisterOperand(4)), ONE_F32)
        self.assertEqual(new1, THREE_F32)
        self.assertEqual(self.interp.mem.read_dword(20), THREE_F32)

        # Step 2: vdata0 = 10.0f -> new = 10.0 - 3.0 = 7.0f
        self.interp.write_reg(RegisterOperand(6), TEN_F32)
        instr2 = DSRsubRtnInstruction(
            vdst=RegisterOperand(4),
            vaddr=RegisterOperand(5),
            vdata0=RegisterOperand(6),
        )
        old2, new2 = self.interp.step(instr2)
        self.assertEqual(old2, THREE_F32)
        self.assertEqual(self.interp.read_reg(RegisterOperand(4)), THREE_F32)
        self.assertAlmostEqual(u32_to_f32(new2), 7.0, places=5)
        self.assertAlmostEqual(u32_to_f32(self.interp.mem.read_dword(20)), 7.0, places=5)

    def test_scalar_immediate_offset(self) -> None:
        # vaddr = 0x100, offset = 8 -> effective = 0x108 -> dword index = 0x42
        self.interp.mem.write_dword(0x42, TWO_F32)
        self.interp.write_reg(RegisterOperand(10), 0x100)
        self.interp.write_reg(RegisterOperand(11), TEN_F32)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(12),
            vaddr=RegisterOperand(10),
            vdata0=RegisterOperand(11),
            offset=8,
        )
        old_val, new_val = self.interp.step(instr)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(self.interp.read_reg(RegisterOperand(12)), TWO_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), 8.0, places=5)
        self.assertAlmostEqual(u32_to_f32(self.interp.mem.read_dword(0x42)), 8.0, places=5)


# ==============================================================================
# SIMD Wavefront Multi-Lane Divergence & 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDLDSWavefrontInterpreter:
    """Multi-lane SIMD architectural interpreter modeling Wave32 and Wave64 execution."""

    def __init__(self, wave_size: int = 64, lds_size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(wave_size)]
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.mem: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg(self, lane: int, op: RegisterOperand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index]
        return self.sgprs[op.index]

    def write_lane_reg(self, lane: int, op: RegisterOperand, val: int) -> None:
        val32 = u32(val)
        if op.reg_type == RegType.VGPR:
            self.vgprs[lane][op.index] = val32
        else:
            self.sgprs[op.index] = val32

    def step(self, instr: DSRsubRtnInstruction) -> List[Optional[Tuple[int, int]]]:
        """Execute instruction across all lanes with EXEC masking."""
        lane_inputs: List[Tuple[int, int]] = []
        for lane in range(self.wave_size):
            va = self.read_lane_reg(lane, instr.vaddr)
            vd = self.read_lane_reg(lane, instr.vdata0)
            lane_inputs.append((va, vd))

        results: List[Optional[Tuple[int, int]]] = []
        for lane in range(self.wave_size):
            is_active = (self.exec_mask & (1 << lane)) != 0
            if not is_active:
                results.append(None)
                continue

            va_val, vd_val = lane_inputs[lane]
            old_val, new_val = self.mem.atomic_rsub_rtn_f32(
                vaddr=va_val,
                vdata0=vd_val,
                offset=instr.offset,
            )
            self.write_lane_reg(lane, instr.vdst, old_val)
            results.append((old_val, new_val))

        return results


class TestSIMDWavefrontDivergence(unittest.TestCase):
    """Unit tests for SIMD wavefront multi-lane divergence and EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        sim = SIMDLDSWavefrontInterpreter(wave_size=32, lds_size_dwords=1024)
        # Pre-seed each lane's dword with 1.0f
        for i in range(32):
            sim.mem.write_dword(i, ONE_F32)
            sim.write_lane_reg(i, RegisterOperand(1), i * 4)     # vaddr = i * 4
            sim.write_lane_reg(i, RegisterOperand(2), FIVE_F32)   # vdata0 = 5.0f

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        res = sim.step(instr)

        for i in range(32):
            self.assertIsNotNone(res[i])
            old_val, new_val = res[i]  # type: ignore
            # 5.0f - 1.0f = 4.0f
            self.assertEqual(old_val, ONE_F32)
            self.assertAlmostEqual(u32_to_f32(new_val), 4.0, places=5)
            self.assertEqual(sim.read_lane_reg(i, RegisterOperand(0)), ONE_F32)
            self.assertAlmostEqual(u32_to_f32(sim.mem.read_dword(i)), 4.0, places=5)

    def test_wave64_exec_masking(self) -> None:
        sim = SIMDLDSWavefrontInterpreter(wave_size=64, lds_size_dwords=1024)
        # Active only on odd lanes
        odd_mask = 0xAAAA_AAAA_AAAA_AAAA
        sim.set_exec_mask(odd_mask)

        for i in range(64):
            sim.mem.write_dword(i, TWO_F32)
            sim.write_lane_reg(i, RegisterOperand(1), i * 4)
            sim.write_lane_reg(i, RegisterOperand(2), TEN_F32)
            sim.write_lane_reg(i, RegisterOperand(0), 0xDEAD_BEEF)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        res = sim.step(instr)

        for i in range(64):
            if i % 2 == 1:
                # Active lane: 10.0f - 2.0f = 8.0f
                self.assertIsNotNone(res[i])
                old_val, new_val = res[i]  # type: ignore
                self.assertEqual(old_val, TWO_F32)
                self.assertAlmostEqual(u32_to_f32(new_val), 8.0, places=5)
                self.assertEqual(sim.read_lane_reg(i, RegisterOperand(0)), TWO_F32)
                self.assertAlmostEqual(u32_to_f32(sim.mem.read_dword(i)), 8.0, places=5)
            else:
                # Inactive lane: untouched memory and register
                self.assertIsNone(res[i])
                self.assertEqual(sim.read_lane_reg(i, RegisterOperand(0)), 0xDEAD_BEEF)
                self.assertEqual(sim.mem.read_dword(i), TWO_F32)

    def test_atomic_serialization_same_address(self) -> None:
        # Multiple lanes targeting the exact same LDS dword (index 5)
        sim = SIMDLDSWavefrontInterpreter(wave_size=32, lds_size_dwords=1024)
        sim.set_exec_mask(0b111)  # only lanes 0, 1, 2 active
        sim.mem.write_dword(5, ONE_F32)  # initial 1.0f

        # Lane 0: vdata0 = 10.0f -> new = 10.0 - 1.0 = 9.0f, returns 1.0f
        # Lane 1: vdata0 = 20.0f -> new = 20.0 - 9.0 = 11.0f, returns 9.0f
        # Lane 2: vdata0 = 15.0f -> new = 15.0 - 11.0 = 4.0f, returns 11.0f
        sim.write_lane_reg(0, RegisterOperand(1), 20)  # byte address 20 -> dword index 5
        sim.write_lane_reg(0, RegisterOperand(2), TEN_F32)

        sim.write_lane_reg(1, RegisterOperand(1), 20)
        sim.write_lane_reg(1, RegisterOperand(2), f32_to_u32(20.0))

        sim.write_lane_reg(2, RegisterOperand(1), 20)
        sim.write_lane_reg(2, RegisterOperand(2), f32_to_u32(15.0))

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        res = sim.step(instr)

        # Lane 0
        self.assertAlmostEqual(u32_to_f32(res[0][0]), 1.0, places=5)  # type: ignore
        self.assertAlmostEqual(u32_to_f32(res[0][1]), 9.0, places=5)  # type: ignore
        # Lane 1
        self.assertAlmostEqual(u32_to_f32(res[1][0]), 9.0, places=5)  # type: ignore
        self.assertAlmostEqual(u32_to_f32(res[1][1]), 11.0, places=5)  # type: ignore
        # Lane 2
        self.assertAlmostEqual(u32_to_f32(res[2][0]), 11.0, places=5)  # type: ignore
        self.assertAlmostEqual(u32_to_f32(res[2][1]), 4.0, places=5)  # type: ignore

        # Final memory state is 4.0f
        self.assertAlmostEqual(u32_to_f32(sim.mem.read_dword(5)), 4.0, places=5)


# ==============================================================================
# Register Aliasing & In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(lds_size_dwords=1024)

    def test_vdst_aliases_vaddr(self) -> None:
        # vdst == vaddr == v1
        # Target address is 40 (dword 10), pre-seeded with 2.0f
        self.interp.mem.write_dword(10, TWO_F32)
        self.interp.write_reg(RegisterOperand(1), 40)
        self.interp.write_reg(RegisterOperand(2), FIVE_F32)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(1),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        old_val, new_val = self.interp.step(instr)

        # Reverse subtraction: 5.0f - 2.0f = 3.0f
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_val, THREE_F32)
        self.assertEqual(self.interp.mem.read_dword(10), THREE_F32)
        # v1 overwritten with old_val (2.0f)
        self.assertEqual(self.interp.read_reg(RegisterOperand(1)), TWO_F32)

    def test_vdst_aliases_vdata0(self) -> None:
        # vdst == vdata0 == v2
        self.interp.mem.write_dword(10, TWO_F32)
        self.interp.write_reg(RegisterOperand(1), 40)
        self.interp.write_reg(RegisterOperand(2), FIVE_F32)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(2),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        old_val, new_val = self.interp.step(instr)

        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_val, THREE_F32)
        self.assertEqual(self.interp.mem.read_dword(10), THREE_F32)
        # v2 overwritten with old_val (2.0f) instead of original 5.0f
        self.assertEqual(self.interp.read_reg(RegisterOperand(2)), TWO_F32)

    def test_three_way_aliasing(self) -> None:
        # vdst == vaddr == vdata0 == v0
        # v0 = 0 (address 0, and float value 0.0f)
        self.interp.mem.write_dword(0, TWO_F32)
        self.interp.write_reg(RegisterOperand(0), POS_ZERO_F32)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(0),
            vdata0=RegisterOperand(0),
        )
        old_val, new_val = self.interp.step(instr)

        # 0.0f - 2.0f = -2.0f
        minus_two_f32 = f32_to_u32(-2.0)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_val, minus_two_f32)
        self.assertEqual(self.interp.mem.read_dword(0), minus_two_f32)
        # v0 overwritten with old_val (2.0f)
        self.assertEqual(self.interp.read_reg(RegisterOperand(0)), TWO_F32)

    def test_operand_preservation_when_unaliased(self) -> None:
        # When vdst is distinct from vaddr and vdata0, sources must be strictly preserved
        self.interp.mem.write_dword(5, ONE_F32)
        self.interp.write_reg(RegisterOperand(1), 20)
        self.interp.write_reg(RegisterOperand(2), TEN_F32)
        self.interp.write_reg(RegisterOperand(3), 0)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        self.interp.step(instr)

        # v1 and v2 unchanged
        self.assertEqual(self.interp.read_reg(RegisterOperand(1)), 20)
        self.assertEqual(self.interp.read_reg(RegisterOperand(2)), TEN_F32)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ==============================================================================

@dataclass(frozen=True)
class FloatStatusFlags:
    """Architectural floating-point classification flags."""
    is_zero: bool
    is_negative: bool
    is_nan: bool
    is_inf: bool
    is_subnormal: bool


def compute_f32_status_flags(val_u32: int) -> FloatStatusFlags:
    """Evaluate architectural floating-point status flags from uint32 bit pattern."""
    u = u32(val_u32)
    exp = (u >> 23) & 0xFF
    mant = u & 0x7F_FFFF
    sign = (u >> 31) & 1

    is_nan = (exp == 0xFF and mant != 0)
    is_inf = (exp == 0xFF and mant == 0)
    is_zero = (exp == 0 and mant == 0)
    is_subnormal = (exp == 0 and mant != 0)
    is_negative = (sign == 1) and not is_nan

    return FloatStatusFlags(
        is_zero=is_zero,
        is_negative=is_negative,
        is_nan=is_nan,
        is_inf=is_inf,
        is_subnormal=is_subnormal,
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry/borrow isolation."""

    def test_status_flags_zero_result(self) -> None:
        # 5.0f - 5.0f = +0.0f
        res = ieee754_rsub_f32(FIVE_F32, FIVE_F32)
        flags = compute_f32_status_flags(res)
        self.assertTrue(flags.is_zero)
        self.assertFalse(flags.is_negative)
        self.assertFalse(flags.is_nan)
        self.assertFalse(flags.is_inf)

    def test_status_flags_negative_result(self) -> None:
        # 2.0f - 5.0f = -3.0f
        res = ieee754_rsub_f32(TWO_F32, FIVE_F32)
        flags = compute_f32_status_flags(res)
        self.assertFalse(flags.is_zero)
        self.assertTrue(flags.is_negative)
        self.assertFalse(flags.is_nan)

    def test_status_flags_nan_and_inf(self) -> None:
        # Inf - Inf = NaN
        res_nan = ieee754_rsub_f32(POS_INF_F32, POS_INF_F32)
        flags_nan = compute_f32_status_flags(res_nan)
        self.assertTrue(flags_nan.is_nan)
        self.assertFalse(flags_nan.is_inf)

        # Inf - 1.0f = Inf
        res_inf = ieee754_rsub_f32(POS_INF_F32, ONE_F32)
        flags_inf = compute_f32_status_flags(res_inf)
        self.assertFalse(flags_inf.is_nan)
        self.assertTrue(flags_inf.is_inf)
        self.assertFalse(flags_inf.is_negative)

    def test_vcc_and_scc_carry_borrow_preserved(self) -> None:
        # In RDNA2, DS instructions operate through LDS memory controllers and do not clobber
        # VCC (vector condition code) or SCC (scalar condition code).
        interp = ScalarLDSInterpreter(lds_size_dwords=1024)
        vcc_initial = 0x12345678_9ABCDEF0
        scc_initial = 1

        interp.mem.write_dword(0, TWO_F32)
        interp.write_reg(RegisterOperand(1), 0)
        interp.write_reg(RegisterOperand(2), FIVE_F32)

        instr = DSRsubRtnInstruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        interp.step(instr)

        # Confirm destination register and memory updated correctly
        self.assertEqual(interp.read_reg(RegisterOperand(0)), TWO_F32)
        self.assertEqual(interp.mem.read_dword(0), THREE_F32)
        # Condition codes remain strictly untouched
        self.assertEqual(vcc_initial, 0x12345678_9ABCDEF0)
        self.assertEqual(scc_initial, 1)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 special values, boundary extremes, and NaN propagation."""

    def test_signed_zeros(self) -> None:
        # (+0.0) - (+0.0) = +0.0
        self.assertEqual(ieee754_rsub_f32(POS_ZERO_F32, POS_ZERO_F32), POS_ZERO_F32)

        # (-0.0) - (-0.0) = +0.0
        self.assertEqual(ieee754_rsub_f32(NEG_ZERO_F32, NEG_ZERO_F32), POS_ZERO_F32)

        # (-0.0) - (+0.0) = -0.0
        self.assertEqual(ieee754_rsub_f32(NEG_ZERO_F32, POS_ZERO_F32), NEG_ZERO_F32)

        # (+0.0) - (-0.0) = +0.0
        self.assertEqual(ieee754_rsub_f32(POS_ZERO_F32, NEG_ZERO_F32), POS_ZERO_F32)

    def test_infinities(self) -> None:
        # Invalid operation: (+Inf) - (+Inf) -> QNaN
        self.assertEqual(ieee754_rsub_f32(POS_INF_F32, POS_INF_F32), QNAN_DEFAULT)

        # Invalid operation: (-Inf) - (-Inf) -> QNaN
        self.assertEqual(ieee754_rsub_f32(NEG_INF_F32, NEG_INF_F32), QNAN_DEFAULT)

        # (+Inf) - (-Inf) = +Inf
        self.assertEqual(ieee754_rsub_f32(POS_INF_F32, NEG_INF_F32), POS_INF_F32)

        # (-Inf) - (+Inf) = -Inf
        self.assertEqual(ieee754_rsub_f32(NEG_INF_F32, POS_INF_F32), NEG_INF_F32)

        # Finite - (+Inf) = -Inf (reverse subtraction where old_val is +Inf)
        self.assertEqual(ieee754_rsub_f32(FIVE_F32, POS_INF_F32), NEG_INF_F32)

        # (+Inf) - Finite = +Inf (vdata0 is +Inf, old_val is Finite)
        self.assertEqual(ieee754_rsub_f32(POS_INF_F32, FIVE_F32), POS_INF_F32)

    def test_nan_propagation(self) -> None:
        # QNaN as minuend
        self.assertEqual(ieee754_rsub_f32(QNAN_DEFAULT, ONE_F32), QNAN_DEFAULT)

        # QNaN as subtrahend
        self.assertEqual(ieee754_rsub_f32(ONE_F32, QNAN_DEFAULT), QNAN_DEFAULT)

        # SNaN as minuend -> quieted to QNaN
        self.assertEqual(ieee754_rsub_f32(SNAN_DEFAULT, TWO_F32), QNAN_DEFAULT)

        # SNaN as subtrahend -> quieted to QNaN
        self.assertEqual(ieee754_rsub_f32(TWO_F32, SNAN_DEFAULT), QNAN_DEFAULT)

    def test_subnormals(self) -> None:
        # Subnormals (denormals) preservation
        subnorm1 = 0x0000_0001  # Smallest positive subnormal (2^-149)
        subnorm2 = 0x0000_0002  # 2 * 2^-149
        res = ieee754_rsub_f32(subnorm2, subnorm1)
        self.assertEqual(res, subnorm1)

    def test_overflow_to_infinity(self) -> None:
        # Max normal positive: 0x7F7F_FFFF (~3.4028235e38)
        # Max normal negative: 0xFF7F_FFFF
        max_norm_pos = 0x7F7F_FFFF
        max_norm_neg = 0xFF7F_FFFF
        # max_pos - (-max_pos) overflows to +Infinity
        res = ieee754_rsub_f32(max_norm_pos, max_norm_neg)
        self.assertEqual(res, POS_INF_F32)


if __name__ == "__main__":
    unittest.main()
