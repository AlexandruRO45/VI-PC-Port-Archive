from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_cmpst_f64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Floating-Point Compare-and-Swap:
Performs an atomic IEEE 754 double-precision float comparison on LDS memory and swaps if equal.

Syntax:
    ds_cmpst_f64 vaddr, vdata0, vdata1 [offset:offset16] [gds]
    ds_cmpst_rtn_f64 vdst, vaddr, vdata0, vdata1 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    addr_lo = effective_byte_address & ~3
    old_raw64 = (LDS[addr_lo + 4] << 32) | LDS[addr_lo]
    if float_equal_f64(old_raw64, vdata0):
        LDS[addr_lo] = vdata1 & 0xFFFFFFFF
        LDS[addr_lo + 4] = (vdata1 >> 32) & 0xFFFFFFFF
    if returning:
        vdst = old_raw64
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000
QNAN_F64: int = 0x7FF8_0000_0000_0000
SNAN_F64: int = 0x7FF0_0000_0000_0001
MIN_NORMAL_F64: int = 0x0010_0000_0000_0000
MIN_SUBNORMAL_F64: int = 0x0000_0000_0000_0001
MAX_SUBNORMAL_F64: int = 0x000F_FFFF_FFFF_FFFF
MAX_NORMAL_F64: int = 0x7FEF_FFFF_FFFF_FFFF
ONE_F64: int = 0x3FF0_0000_0000_0000


def u32(val: int) -> int:
    return val & MASK32


def u64(val: int) -> int:
    return val & MASK64


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_f64_nan(bits: int) -> bool:
    return (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def is_f64_subnormal(bits: int) -> bool:
    return (bits & 0x7FF0_0000_0000_0000) == 0 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def float_equal_f64(a_bits: int, b_bits: int) -> bool:
    if is_f64_nan(a_bits) or is_f64_nan(b_bits):
        return False
    return bits_to_f64(a_bits) == bits_to_f64(b_bits)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    return (eff_addr >> 2) & (capacity_dwords - 1)


def ref_ds_cmpst_f64(
    current_mem: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_cmpst_f64.

    Compares current_mem with cmp_val using IEEE 754 double-precision float equality;
    if equal, replaces memory with new_val.
    Always returns (old_mem64, new_mem64, swapped).
    """
    old_raw = u64(current_mem)
    cmp_raw = u64(cmp_val)
    new_raw = u64(new_val)

    if float_equal_f64(old_raw, cmp_raw):
        return old_raw, new_raw, True
    return old_raw, old_raw, False


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_qword(self, eff_addr: int) -> int:
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        return combine_u32s_to_u64(self.memory[idx_lo], self.memory[idx_hi])

    def write_qword(self, eff_addr: int, val64: int) -> None:
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        lo, hi = split_u64_to_u32s(val64)
        self.memory[idx_lo] = lo
        self.memory[idx_hi] = hi

    def atomic_cmpst_f64(
        self,
        vaddr: int,
        cmp_val64: int,
        new_val64: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem, swapped = ref_ds_cmpst_f64(curr, cmp_val64, new_val64)
        if swapped:
            self.write_qword(eff_addr, new_mem)
        return old_val, new_mem, swapped



# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    DS_CMPST_F64 = "ds_cmpst_f64"
    DS_CMPST_RTN_F64 = "ds_cmpst_rtn_f64"


class RegType(Enum):
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegSingle:
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(f"Register index out of bounds: {self.reg_type.value}[{self.index}]")

    def __str__(self) -> str:
        return f"{self.reg_type.value}[{self.index}]"


@dataclass(frozen=True)
class RegPair:
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = (DEFAULT_NUM_VGPRS - 1) if self.reg_type == RegType.VGPR else (DEFAULT_NUM_SGPRS - 1)
        if not (0 <= self.index < max_limit):
            raise ValueError(f"Register pair base index out of bounds: {self.reg_type.value}[{self.index}:{self.index + 1}]")

    def __str__(self) -> str:
        return f"{self.reg_type.value}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class ImmediateF64:
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u64(self.val))

    def __str__(self) -> str:
        return f"0x{self.val:X}"


Operand64 = Union[RegPair, ImmediateF64]


@dataclass(frozen=True)
class DSCmpstF64Instruction:
    """Symbolic AST node for ds_cmpst_f64 / ds_cmpst_rtn_f64 instruction."""
    vaddr: RegSingle
    vdata0: Operand64  # Double compare value
    vdata1: Operand64  # Double new value
    vdst: Optional[RegPair] = None  # None for ds_cmpst_f64, present for ds_cmpst_rtn_f64
    offset: int = 0
    gds: bool = False

    @property
    def opcode(self) -> DSOpcode:
        return DSOpcode.DS_CMPST_RTN_F64 if self.vdst is not None else DSOpcode.DS_CMPST_F64

    def __post_init__(self) -> None:
        if not isinstance(self.vaddr, RegSingle) or self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR RegSingle, got {self.vaddr}")
        if self.vdst is not None and (not isinstance(self.vdst, RegPair) or self.vdst.reg_type != RegType.VGPR):
            raise ValueError(f"vdst must be a VGPR RegPair, got {self.vdst}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of bounds: {self.offset}")

    def disassemble(self) -> str:
        parts = [self.opcode.value]
        if self.vdst is not None:
            parts.append(str(self.vdst))
        parts.extend([str(self.vaddr), str(self.vdata0), str(self.vdata1)])
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return f"{parts[0]} {', '.join(parts[1:])}"



class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_finite_double_match(self) -> None:
        val = f64_to_bits(3.141592653589793)
        new_val = f64_to_bits(2.718281828459045)
        old, mem, swapped = ref_ds_cmpst_f64(val, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)

    def test_oracle_signed_zero_equality(self) -> None:
        # In IEEE 754, +0.0 == -0.0 is True!
        old, mem, swapped = ref_ds_cmpst_f64(NEG_ZERO_F64, POS_ZERO_F64, f64_to_bits(100.0))
        self.assertTrue(swapped)
        self.assertEqual(old, NEG_ZERO_F64)
        self.assertEqual(mem, f64_to_bits(100.0))

    def test_oracle_nan_unordered(self) -> None:
        old, mem, swapped = ref_ds_cmpst_f64(QNAN_F64, QNAN_F64, f64_to_bits(1.0))
        self.assertFalse(swapped)
        self.assertEqual(mem, QNAN_F64)

    def test_oracle_infinities(self) -> None:
        old, mem, swapped = ref_ds_cmpst_f64(POS_INF_F64, POS_INF_F64, 0)
        self.assertTrue(swapped)
        self.assertEqual(mem, 0)

        old2, mem2, swapped2 = ref_ds_cmpst_f64(POS_INF_F64, NEG_INF_F64, 0)
        self.assertFalse(swapped2)
        self.assertEqual(mem2, POS_INF_F64)

    def test_architectural_memory_atomic_cmpst_f64(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x80
        val = f64_to_bits(42.0)
        new_val = f64_to_bits(84.0)

        lds.write_qword(addr, val)
        old, mem, swapped = lds.atomic_cmpst_f64(addr, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_qword(addr), new_val)



class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_non_returning_ast(self) -> None:
        vaddr = RegSingle(1)
        vdata0 = RegPair(2)
        vdata1 = RegPair(4)
        instr = DSCmpstF64Instruction(vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=16)

        self.assertEqual(instr.opcode, DSOpcode.DS_CMPST_F64)
        self.assertIsNone(instr.vdst)
        self.assertEqual(instr.disassemble(), "ds_cmpst_f64 v[1], v[2:3], v[4:5], offset:16")

    def test_returning_ast(self) -> None:
        vdst = RegPair(0)
        vaddr = RegSingle(2)
        vdata0 = ImmediateF64(f64_to_bits(1.0))
        vdata1 = ImmediateF64(f64_to_bits(2.0))
        instr = DSCmpstF64Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1)

        self.assertEqual(instr.opcode, DSOpcode.DS_CMPST_RTN_F64)
        self.assertEqual(instr.disassemble(), "ds_cmpst_rtn_f64 v[0:1], v[2], 0x3FF0000000000000, 0x4000000000000000")

    def test_ast_validation_errors(self) -> None:
        v0 = RegPair(0)
        v1 = RegSingle(1)
        s0 = RegSingle(0, RegType.SGPR)

        with self.assertRaises(ValueError):
            DSCmpstF64Instruction(vaddr=s0, vdata0=v0, vdata1=v0)
        with self.assertRaises(ValueError):
            DSCmpstF64Instruction(vaddr=v1, vdata0=v0, vdata1=v0, offset=-1)



if __name__ == "__main__":
    unittest.main()
