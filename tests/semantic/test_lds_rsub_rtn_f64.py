from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_rsub_rtn_f64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Floating-Point Reverse Subtraction with Return:
Performs an atomic double-precision IEEE 754-2008/2019 floating-point reverse subtraction on 64-bit LDS memory
and returns the original (pre-operation) 64-bit value:
    new_val = vdata0 - old_val
    vdst = old_val

Syntax:
    ds_rsub_rtn_f64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val_u64 = LDS64[qword_index]
    new_val_u64 = ieee754_rsub_f64(vdata0, old_val_u64)  # vdata0 is minuend, old_val is subtrahend
    LDS64[qword_index] = new_val_u64
    vdst = old_val_u64
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_QWORDS: int = 8192         # 8,192 64-bit qwords
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Binary64 Architectural Bitmasks & Constants
SIGN_MASK64: int = 0x8000_0000_0000_0000
EXP_MASK64: int = 0x7FF0_0000_0000_0000
FRAC_MASK64: int = 0x000F_FFFF_FFFF_FFFF
CANONICAL_QNAN64: int = 0x7FF8_0000_0000_0000
CANONICAL_SNAN64: int = 0x7FF0_0000_0000_0001
POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000

# Convenient float64 constants (raw uint64 bit patterns)
ONE_F64: int = 0x3FF0_0000_0000_0000        # 1.0
TWO_F64: int = 0x4000_0000_0000_0000        # 2.0
THREE_F64: int = 0x4008_0000_0000_0000      # 3.0
FIVE_F64: int = 0x4014_0000_0000_0000       # 5.0
TEN_F64: int = 0x4024_0000_0000_0000        # 10.0


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def u64_to_f64(val: int) -> float:
    """Unpack 64-bit unsigned qword to Python float."""
    return struct.unpack("<d", struct.pack("<Q", u64(val)))[0]


def f64_to_u64(val: float) -> int:
    """Pack Python float to IEEE 754 binary64 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return NEG_INF_F64 if val < 0 else POS_INF_F64


def ieee754_rsub_f64(vdata0_u64: int, old_val_u64: int) -> int:
    """Bit-accurate IEEE 754 double-precision reverse subtraction: vdata0 - old_val.

    Matches RDNA2 hardware semantics where vdata0 is minuend and old_val is subtrahend.
    """
    a_u64 = u64(vdata0_u64)
    b_u64 = u64(old_val_u64)

    a_exp = (a_u64 & EXP_MASK64) >> 52
    a_frac = a_u64 & FRAC_MASK64
    b_exp = (b_u64 & EXP_MASK64) >> 52
    b_frac = b_u64 & FRAC_MASK64

    a_is_nan = (a_exp == 0x7FF and a_frac != 0)
    b_is_nan = (b_exp == 0x7FF and b_frac != 0)

    # Any NaN operand produces canonical quiet NaN
    if a_is_nan or b_is_nan:
        return CANONICAL_QNAN64

    a_is_inf = (a_exp == 0x7FF and a_frac == 0)
    b_is_inf = (b_exp == 0x7FF and b_frac == 0)
    a_sign = (a_u64 >> 63) & 1
    b_sign = (b_u64 >> 63) & 1

    # (+Inf) - (+Inf) or (-Inf) - (-Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign == b_sign):
        return CANONICAL_QNAN64

    # If minuend is Inf, result is minuend
    if a_is_inf:
        return a_u64

    # If subtrahend is Inf, result is -subtrahend
    if b_is_inf:
        return b_u64 ^ SIGN_MASK64

    a_f = u64_to_f64(a_u64)
    b_f = u64_to_f64(b_u64)

    # Signed zeros:
    # (+0.0) - (+0.0) = +0.0
    # (-0.0) - (-0.0) = +0.0
    # (+0.0) - (-0.0) = +0.0
    # (-0.0) - (+0.0) = -0.0
    if a_f == 0.0 and b_f == 0.0:
        if a_sign == 1 and b_sign == 0:
            return NEG_ZERO_F64
        return POS_ZERO_F64

    res_f = a_f - b_f
    res_u64 = f64_to_u64(res_f)

    # In round-to-nearest-even, if difference of equal non-zero numbers is zero, sign is +0.0
    if a_f == b_f and res_f == 0.0:
        return POS_ZERO_F64

    return res_u64


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert effective byte address to aligned LDS 64-bit qword index."""
    return (eff_addr >> 3) & (capacity_qwords - 1)


class LDSMemory64:
    """Architectural model of Local Data Share (LDS) 64-bit memory."""

    def __init__(self, size_qwords: int = LDS_SIZE_QWORDS) -> None:
        if size_qwords <= 0 or (size_qwords & (size_qwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_qwords}")
        self.size_qwords = size_qwords
        self.data: List[int] = [0] * size_qwords

    def read_qword(self, index: int) -> int:
        return self.data[index & (self.size_qwords - 1)]

    def write_qword(self, index: int, val: int) -> None:
        self.data[index & (self.size_qwords - 1)] = u64(val)

    def atomic_rsub_rtn_f64(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit float reverse subtraction on LDS with return.

        Returns:
            Tuple[int, int]: (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.size_qwords)
        old_val = self.read_qword(idx)
        new_val = ieee754_rsub_f64(vdata0, old_val)
        self.write_qword(idx, new_val)
        return old_val, new_val


def oracle_ds_rsub_rtn_f64(
    lds_bytes: bytearray,
    vaddr: int,
    vdata0: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_SIZE_BYTES,
) -> Tuple[int, int]:
    """Pure bytearray reference mathematical oracle for RDNA2 ds_rsub_rtn_f64."""
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~7  # drop low 3 bits (8-byte alignment)
    old_val = struct.unpack_from("<Q", lds_bytes, aligned_addr)[0]
    new_val = ieee754_rsub_f64(vdata0, old_val)
    struct.pack_into("<Q", lds_bytes, aligned_addr, new_val)
    return old_val, new_val


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory64(size_qwords=1024)
        self.lds_bytes = bytearray(LDS_SIZE_BYTES)

    def test_basic_reverse_subtraction(self) -> None:
        # Initial memory has 2.0, vdata0 is 5.0
        # Reverse subtraction: new_val = vdata0 - old_val = 5.0 - 2.0 = 3.0
        # Returns old_val = 2.0
        self.mem.write_qword(5, TWO_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=40, vdata0=FIVE_F64, offset=0)
        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(new_val, THREE_F64)
        self.assertAlmostEqual(u64_to_f64(self.mem.read_qword(5)), 3.0, places=9)

    def test_reverse_subtraction_negative_result(self) -> None:
        # Initial memory has 10.0, vdata0 is 3.0
        # Reverse subtraction: 3.0 - 10.0 = -7.0
        self.mem.write_qword(12, TEN_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=96, vdata0=THREE_F64, offset=0)
        self.assertEqual(old_val, TEN_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), -7.0, places=9)
        self.assertAlmostEqual(u64_to_f64(self.mem.read_qword(12)), -7.0, places=9)

    def test_non_commutativity_difference_from_sub(self) -> None:
        # vdata0 = 2.0, old_val = 5.0
        # rsub: 2.0 - 5.0 = -3.0
        # sub:  5.0 - 2.0 = +3.0
        self.mem.write_qword(15, FIVE_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=120, vdata0=TWO_F64, offset=0)
        self.assertEqual(old_val, FIVE_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), -3.0, places=9)

    def test_address_offset_and_qword_alignment(self) -> None:
        # vaddr = 0x200, offset = 16 -> effective = 0x210 -> qword index = 0x210 >> 3 = 0x42
        idx = 0x42
        self.mem.write_qword(idx, ONE_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=0x200, vdata0=FIVE_F64, offset=16)
        self.assertEqual(old_val, ONE_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), 4.0, places=9)

        # Unaligned byte address: 0x207 drops low 3 bits -> 0x200 (index 0x40)
        self.mem.write_qword(0x40, TWO_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=0x200, vdata0=TEN_F64, offset=7)
        self.assertEqual(old_val, TWO_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), 8.0, places=9)

    def test_bytearray_oracle_equivalence(self) -> None:
        addr = 0x800
        struct.pack_into("<Q", self.lds_bytes, addr, TWO_F64)
        old_val, new_val = oracle_ds_rsub_rtn_f64(self.lds_bytes, vaddr=addr, vdata0=FIVE_F64, offset=0)
        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(new_val, THREE_F64)
        self.assertEqual(struct.unpack_from("<Q", self.lds_bytes, addr)[0], THREE_F64)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Strongly-typed scalar or vector register operand."""
    index: int
    reg_type: RegType = RegType.VGPR
    count: int = 1

    def __post_init__(self) -> None:
        if self.count not in (1, 2):
            raise ValueError(f"Register count must be 1 (32-bit) or 2 (64-bit), got {self.count}")
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index and self.index + self.count <= DEFAULT_NUM_VGPRS):
                raise ValueError(f"VGPR index/range out of bounds [0, {DEFAULT_NUM_VGPRS}]: index={self.index}, count={self.count}")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index and self.index + self.count <= DEFAULT_NUM_SGPRS):
                raise ValueError(f"SGPR index/range out of bounds [0, {DEFAULT_NUM_SGPRS}]: index={self.index}, count={self.count}")

    @property
    def name(self) -> str:
        if self.count == 2:
            return f"{self.reg_type.value}[{self.index}:{self.index + 1}]"
        return f"{self.reg_type.value}{self.index}"


class DSOpcode(Enum):
    """LDS DS-encoding opcode identifier."""
    DS_RSUB_RTN_F64 = "ds_rsub_rtn_f64"


@dataclass(frozen=True)
class DSRsubRtnF64Instruction:
    """Symbolic AST node representing a ds_rsub_rtn_f64 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: RegisterOperand
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_RSUB_RTN_F64

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR or self.vdst.count != 2:
            raise TypeError(f"vdst must be a 64-bit VGPR pair (count=2), got {self.vdst}")
        if self.vaddr.reg_type != RegType.VGPR or self.vaddr.count != 1:
            raise TypeError(f"vaddr must be a single 32-bit VGPR (count=1), got {self.vaddr}")
        if self.vdata0.reg_type != RegType.VGPR or self.vdata0.count != 2:
            raise TypeError(f"vdata0 must be a 64-bit VGPR pair (count=2), got {self.vdata0}")
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
        self.assertEqual(DSOpcode.DS_RSUB_RTN_F64.value, "ds_rsub_rtn_f64")

    def test_register_operands(self) -> None:
        v0_single = RegisterOperand(0, RegType.VGPR, count=1)
        v0_pair = RegisterOperand(0, RegType.VGPR, count=2)
        s0_single = RegisterOperand(0, RegType.SGPR, count=1)

        self.assertEqual(v0_single.name, "v0")
        self.assertEqual(v0_pair.name, "v[0:1]")
        self.assertEqual(s0_single.name, "s0")

    def test_ast_construction_and_disassembly(self) -> None:
        instr_basic = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
            offset=0,
        )
        self.assertEqual(instr_basic.disasm(), "ds_rsub_rtn_f64 v[0:1], v2, v[4:5]")

        instr_offset = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
            offset=32,
        )
        self.assertEqual(instr_offset.disasm(), "ds_rsub_rtn_f64 v[0:1], v2, v[4:5] offset:32")

        instr_gds = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
            offset=16,
            gds=True,
        )
        self.assertEqual(instr_gds.disasm(), "ds_rsub_rtn_f64 v[0:1], v2, v[4:5] offset:16 gds")

    def test_ast_immutability_and_equality(self) -> None:
        instr1 = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(2, count=2),
            vaddr=RegisterOperand(4, count=1),
            vdata0=RegisterOperand(6, count=2),
            offset=64,
        )
        instr2 = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(2, count=2),
            vaddr=RegisterOperand(4, count=1),
            vdata0=RegisterOperand(6, count=2),
            offset=64,
        )
        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))

    def test_ast_validation_and_type_checks(self) -> None:
        # Invalid register counts
        with self.assertRaises(TypeError):
            DSRsubRtnF64Instruction(
                vdst=RegisterOperand(0, count=1),  # must be count=2
                vaddr=RegisterOperand(2, count=1),
                vdata0=RegisterOperand(4, count=2),
            )
        with self.assertRaises(TypeError):
            DSRsubRtnF64Instruction(
                vdst=RegisterOperand(0, count=2),
                vaddr=RegisterOperand(2, count=2),  # must be count=1
                vdata0=RegisterOperand(4, count=2),
            )
        with self.assertRaises(TypeError):
            DSRsubRtnF64Instruction(
                vdst=RegisterOperand(0, count=2),
                vaddr=RegisterOperand(2, count=1),
                vdata0=RegisterOperand(4, count=1),  # must be count=2
            )

        # SGPR instead of VGPR
        with self.assertRaises(TypeError):
            DSRsubRtnF64Instruction(
                vdst=RegisterOperand(0, RegType.SGPR, count=2),
                vaddr=RegisterOperand(2, count=1),
                vdata0=RegisterOperand(4, count=2),
            )

        # Offset out of range
        with self.assertRaises(ValueError):
            DSRsubRtnF64Instruction(
                vdst=RegisterOperand(0, count=2),
                vaddr=RegisterOperand(2, count=1),
                vdata0=RegisterOperand(4, count=2),
                offset=-1,
            )
        with self.assertRaises(ValueError):
            DSRsubRtnF64Instruction(
                vdst=RegisterOperand(0, count=2),
                vaddr=RegisterOperand(2, count=1),
                vdata0=RegisterOperand(4, count=2),
                offset=65536,
            )


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarLDSInterpreter64:
    """Deterministic scalar architectural interpreter for 64-bit LDS instructions."""

    def __init__(self, lds_size_qwords: int = LDS_SIZE_QWORDS) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.mem: LDSMemory64 = LDSMemory64(size_qwords=lds_size_qwords)

    def read_reg_u32(self, op: RegisterOperand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index]
        return self.sgprs[op.index]

    def read_reg_u64(self, op: RegisterOperand) -> int:
        idx = op.index
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[idx]
            hi = self.vgprs[idx + 1]
            return u64((hi << 32) | lo)
        lo = self.sgprs[idx]
        hi = self.sgprs[idx + 1]
        return u64((hi << 32) | lo)

    def write_reg_u32(self, op: RegisterOperand, val: int) -> None:
        val32 = u32(val)
        if op.reg_type == RegType.VGPR:
            self.vgprs[op.index] = val32
        else:
            self.sgprs[op.index] = val32

    def write_reg_u64(self, op: RegisterOperand, val: int) -> None:
        val64 = u64(val)
        lo = val64 & MASK32
        hi = (val64 >> 32) & MASK32
        idx = op.index
        if op.reg_type == RegType.VGPR:
            self.vgprs[idx] = lo
            self.vgprs[idx + 1] = hi
        else:
            self.sgprs[idx] = lo
            self.sgprs[idx + 1] = hi

    def step(self, instr: DSRsubRtnF64Instruction) -> Tuple[int, int]:
        """Execute a single ds_rsub_rtn_f64 instruction deterministically."""
        vaddr_val = self.read_reg_u32(instr.vaddr)
        vdata0_val = self.read_reg_u64(instr.vdata0)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(
            vaddr=vaddr_val,
            vdata0=vdata0_val,
            offset=instr.offset,
        )
        self.write_reg_u64(instr.vdst, old_val)
        return old_val, new_val


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter64(lds_size_qwords=1024)

    def test_scalar_rsub_step_execution(self) -> None:
        # Pre-seed memory qword index 5 (byte address 40) with 2.0
        self.interp.mem.write_qword(5, TWO_F64)
        # v2 (vaddr) = 40
        self.interp.write_reg_u32(RegisterOperand(2, count=1), 40)
        # v[4:5] (vdata0) = 5.0
        self.interp.write_reg_u64(RegisterOperand(4, count=2), FIVE_F64)

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
            offset=0,
        )
        old_val, new_val = self.interp.step(instr)

        # Pre-op memory value was 2.0 -> returned into v[0:1]
        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(self.interp.read_reg_u64(RegisterOperand(0, count=2)), TWO_F64)
        # Post-op memory value is 5.0 - 2.0 = 3.0
        self.assertEqual(new_val, THREE_F64)
        self.assertEqual(self.interp.mem.read_qword(5), THREE_F64)

    def test_scalar_multi_step_sequence(self) -> None:
        # Pre-seed memory index 10 (byte address 80) with 1.0
        self.interp.mem.write_qword(10, ONE_F64)
        self.interp.write_reg_u32(RegisterOperand(10, count=1), 80)

        # Step 1: vdata0 = 4.0 -> new = 4.0 - 1.0 = 3.0
        four_f64 = f64_to_u64(4.0)
        self.interp.write_reg_u64(RegisterOperand(12, count=2), four_f64)
        instr1 = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(8, count=2),
            vaddr=RegisterOperand(10, count=1),
            vdata0=RegisterOperand(12, count=2),
        )
        old1, new1 = self.interp.step(instr1)
        self.assertEqual(old1, ONE_F64)
        self.assertEqual(self.interp.read_reg_u64(RegisterOperand(8, count=2)), ONE_F64)
        self.assertEqual(new1, THREE_F64)
        self.assertEqual(self.interp.mem.read_qword(10), THREE_F64)

        # Step 2: vdata0 = 10.0 -> new = 10.0 - 3.0 = 7.0
        self.interp.write_reg_u64(RegisterOperand(12, count=2), TEN_F64)
        instr2 = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(8, count=2),
            vaddr=RegisterOperand(10, count=1),
            vdata0=RegisterOperand(12, count=2),
        )
        old2, new2 = self.interp.step(instr2)
        self.assertEqual(old2, THREE_F64)
        self.assertEqual(self.interp.read_reg_u64(RegisterOperand(8, count=2)), THREE_F64)
        self.assertAlmostEqual(u64_to_f64(new2), 7.0, places=9)
        self.assertAlmostEqual(u64_to_f64(self.interp.mem.read_qword(10)), 7.0, places=9)

    def test_scalar_immediate_offset(self) -> None:
        # vaddr = 0x100, offset = 16 -> effective = 0x110 -> qword index = 0x110 >> 3 = 0x22
        self.interp.mem.write_qword(0x22, TWO_F64)
        self.interp.write_reg_u32(RegisterOperand(6, count=1), 0x100)
        self.interp.write_reg_u64(RegisterOperand(8, count=2), TEN_F64)

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(4, count=2),
            vaddr=RegisterOperand(6, count=1),
            vdata0=RegisterOperand(8, count=2),
            offset=16,
        )
        old_val, new_val = self.interp.step(instr)
        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(self.interp.read_reg_u64(RegisterOperand(4, count=2)), TWO_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), 8.0, places=9)
        self.assertAlmostEqual(u64_to_f64(self.interp.mem.read_qword(0x22)), 8.0, places=9)


# ==============================================================================
# SIMD Wavefront Multi-Lane Divergence & 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDLDSWavefrontInterpreter64:
    """Multi-lane SIMD architectural interpreter modeling Wave32 and Wave64 execution for 64-bit LDS."""

    def __init__(self, wave_size: int = 64, lds_size_qwords: int = LDS_SIZE_QWORDS) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(wave_size)]
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.mem: LDSMemory64 = LDSMemory64(size_qwords=lds_size_qwords)

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, op: RegisterOperand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index]
        return self.sgprs[op.index]

    def read_lane_reg_u64(self, lane: int, op: RegisterOperand) -> int:
        idx = op.index
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[lane][idx]
            hi = self.vgprs[lane][idx + 1]
            return u64((hi << 32) | lo)
        lo = self.sgprs[idx]
        hi = self.sgprs[idx + 1]
        return u64((hi << 32) | lo)

    def write_lane_reg_u64(self, lane: int, op: RegisterOperand, val: int) -> None:
        val64 = u64(val)
        lo = val64 & MASK32
        hi = (val64 >> 32) & MASK32
        idx = op.index
        if op.reg_type == RegType.VGPR:
            self.vgprs[lane][idx] = lo
            self.vgprs[lane][idx + 1] = hi
        else:
            self.sgprs[idx] = lo
            self.sgprs[idx + 1] = hi

    def step(self, instr: DSRsubRtnF64Instruction) -> List[Optional[Tuple[int, int]]]:
        """Execute instruction across all lanes with EXEC masking."""
        lane_inputs: List[Tuple[int, int]] = []
        for lane in range(self.wave_size):
            va = self.read_lane_reg_u32(lane, instr.vaddr)
            vd = self.read_lane_reg_u64(lane, instr.vdata0)
            lane_inputs.append((va, vd))

        results: List[Optional[Tuple[int, int]]] = []
        for lane in range(self.wave_size):
            is_active = (self.exec_mask & (1 << lane)) != 0
            if not is_active:
                results.append(None)
                continue

            va_val, vd_val = lane_inputs[lane]
            old_val, new_val = self.mem.atomic_rsub_rtn_f64(
                vaddr=va_val,
                vdata0=vd_val,
                offset=instr.offset,
            )
            self.write_lane_reg_u64(lane, instr.vdst, old_val)
            results.append((old_val, new_val))

        return results


class TestSIMDWavefrontDivergence(unittest.TestCase):
    """Unit tests for SIMD wavefront multi-lane divergence and EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        sim = SIMDLDSWavefrontInterpreter64(wave_size=32, lds_size_qwords=1024)
        # Pre-seed each lane's qword with 1.0
        for i in range(32):
            sim.mem.write_qword(i, ONE_F64)
            sim.vgprs[i][2] = i * 8  # vaddr = i * 8
            # vdata0 = v[4:5] = 5.0
            sim.write_lane_reg_u64(i, RegisterOperand(4, count=2), FIVE_F64)

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )
        res = sim.step(instr)

        for i in range(32):
            self.assertIsNotNone(res[i])
            old_val, new_val = res[i]  # type: ignore
            # 5.0 - 1.0 = 4.0
            self.assertEqual(old_val, ONE_F64)
            self.assertAlmostEqual(u64_to_f64(new_val), 4.0, places=9)
            self.assertEqual(sim.read_lane_reg_u64(i, RegisterOperand(0, count=2)), ONE_F64)
            self.assertAlmostEqual(u64_to_f64(sim.mem.read_qword(i)), 4.0, places=9)

    def test_wave64_exec_masking(self) -> None:
        sim = SIMDLDSWavefrontInterpreter64(wave_size=64, lds_size_qwords=1024)
        odd_mask = 0xAAAA_AAAA_AAAA_AAAA
        sim.set_exec_mask(odd_mask)

        for i in range(64):
            sim.mem.write_qword(i, TWO_F64)
            sim.vgprs[i][2] = i * 8
            sim.write_lane_reg_u64(i, RegisterOperand(4, count=2), TEN_F64)
            sim.write_lane_reg_u64(i, RegisterOperand(0, count=2), 0xDEAD_BEEF_CAFE_BABE)

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )
        res = sim.step(instr)

        for i in range(64):
            if i % 2 == 1:
                # Active lane: 10.0 - 2.0 = 8.0
                self.assertIsNotNone(res[i])
                old_val, new_val = res[i]  # type: ignore
                self.assertEqual(old_val, TWO_F64)
                self.assertAlmostEqual(u64_to_f64(new_val), 8.0, places=9)
                self.assertEqual(sim.read_lane_reg_u64(i, RegisterOperand(0, count=2)), TWO_F64)
                self.assertAlmostEqual(u64_to_f64(sim.mem.read_qword(i)), 8.0, places=9)
            else:
                # Inactive lane: untouched memory and register
                self.assertIsNone(res[i])
                self.assertEqual(sim.read_lane_reg_u64(i, RegisterOperand(0, count=2)), 0xDEAD_BEEF_CAFE_BABE)
                self.assertEqual(sim.mem.read_qword(i), TWO_F64)

    def test_atomic_serialization_same_address(self) -> None:
        sim = SIMDLDSWavefrontInterpreter64(wave_size=32, lds_size_qwords=1024)
        sim.set_exec_mask(0b111)  # only lanes 0, 1, 2
        sim.mem.write_qword(3, ONE_F64)  # initial 1.0 at qword 3 (byte address 24)

        # Lane 0: vdata0 = 10.0 -> new = 10.0 - 1.0 = 9.0, returns 1.0
        # Lane 1: vdata0 = 20.0 -> new = 20.0 - 9.0 = 11.0, returns 9.0
        # Lane 2: vdata0 = 15.0 -> new = 15.0 - 11.0 = 4.0, returns 11.0
        sim.vgprs[0][2] = 24
        sim.write_lane_reg_u64(0, RegisterOperand(4, count=2), TEN_F64)

        sim.vgprs[1][2] = 24
        sim.write_lane_reg_u64(1, RegisterOperand(4, count=2), f64_to_u64(20.0))

        sim.vgprs[2][2] = 24
        sim.write_lane_reg_u64(2, RegisterOperand(4, count=2), f64_to_u64(15.0))

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )
        res = sim.step(instr)

        self.assertAlmostEqual(u64_to_f64(res[0][0]), 1.0, places=9)  # type: ignore
        self.assertAlmostEqual(u64_to_f64(res[0][1]), 9.0, places=9)  # type: ignore
        self.assertAlmostEqual(u64_to_f64(res[1][0]), 9.0, places=9)  # type: ignore
        self.assertAlmostEqual(u64_to_f64(res[1][1]), 11.0, places=9)  # type: ignore
        self.assertAlmostEqual(u64_to_f64(res[2][0]), 11.0, places=9)  # type: ignore
        self.assertAlmostEqual(u64_to_f64(res[2][1]), 4.0, places=9)  # type: ignore

        self.assertAlmostEqual(u64_to_f64(sim.mem.read_qword(3)), 4.0, places=9)


# ==============================================================================
# Register Aliasing & In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter64(lds_size_qwords=1024)

    def test_vdst_aliases_vdata0(self) -> None:
        # vdst == vdata0 == v[0:1]
        # Target address is 40 (qword 5), pre-seeded with 2.0
        self.interp.mem.write_qword(5, TWO_F64)
        self.interp.write_reg_u32(RegisterOperand(2, count=1), 40)
        self.interp.write_reg_u64(RegisterOperand(0, count=2), FIVE_F64)

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(0, count=2),
        )
        old_val, new_val = self.interp.step(instr)

        # Reverse subtraction: 5.0 - 2.0 = 3.0
        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(new_val, THREE_F64)
        self.assertEqual(self.interp.mem.read_qword(5), THREE_F64)
        # v[0:1] overwritten with old_val (2.0)
        self.assertEqual(self.interp.read_reg_u64(RegisterOperand(0, count=2)), TWO_F64)

    def test_vdst_partially_aliases_vaddr(self) -> None:
        # vdst is v[0:1], vaddr is v0 (byte address 0)
        self.interp.mem.write_qword(0, TWO_F64)
        self.interp.write_reg_u32(RegisterOperand(0, count=1), 0)
        self.interp.write_reg_u64(RegisterOperand(2, count=2), FIVE_F64)

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(0, count=1),
            vdata0=RegisterOperand(2, count=2),
        )
        old_val, new_val = self.interp.step(instr)

        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(new_val, THREE_F64)
        self.assertEqual(self.interp.mem.read_qword(0), THREE_F64)
        # v[0:1] receives old_val
        self.assertEqual(self.interp.read_reg_u64(RegisterOperand(0, count=2)), TWO_F64)

    def test_operand_preservation_when_unaliased(self) -> None:
        self.interp.mem.write_qword(2, ONE_F64)
        self.interp.write_reg_u32(RegisterOperand(4, count=1), 16)
        self.interp.write_reg_u64(RegisterOperand(6, count=2), TEN_F64)
        self.interp.write_reg_u64(RegisterOperand(0, count=2), 0)

        instr = DSRsubRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(4, count=1),
            vdata0=RegisterOperand(6, count=2),
        )
        self.interp.step(instr)

        # Sources preserved
        self.assertEqual(self.interp.read_reg_u32(RegisterOperand(4, count=1)), 16)
        self.assertEqual(self.interp.read_reg_u64(RegisterOperand(6, count=2)), TEN_F64)


if __name__ == "__main__":
    unittest.main()
