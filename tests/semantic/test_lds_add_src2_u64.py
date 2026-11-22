from __future__ import annotations

"""Reference oracle and architectural model for ds_add_src2_u64.

RDNA2 / GFX10 Local Data Share (LDS) 64-bit Dual-Source Atomic Addition:
Performs a 64-bit atomic addition on LDS memory location A using
the secondary LDS memory location B as the addend: LDS[A] = (LDS[A] + LDS[B]) mod 2^64.

Syntax:
    ds_add_src2_u64 vaddr [offset0:offset8] [offset1:offset8] [gds]

Operands:
    vaddr: VGPR containing base byte address into LDS.
    offset0: 8-bit unsigned immediate multiplied by 8 bytes (location A stride).
    offset1: 8-bit unsigned immediate multiplied by 8 bytes (location B stride).

Semantics:
    byte_addr_a = (vaddr + (offset0 & 0xFF) * 8) & 0xFFFFFFFF
    byte_addr_b = (vaddr + (offset1 & 0xFF) * 8) & 0xFFFFFFFF
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    val_a = LDS64[qword_idx_a]
    val_b = LDS64[qword_idx_b]
    LDS64[qword_idx_a] = (val_a + val_b) & 0xFFFFFFFFFFFFFFFF
    # Location B remains unchanged (unless qword_idx_a == qword_idx_b)
    # Non-returning: VGPRs are not modified.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
OFFSET8_MASK: int = 0xFF
DEFAULT_LDS_QWORDS: int = 8192     # 64 KiB default capacity (8192 * 8 bytes)
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit qword into (low_u32, high_u32)."""
    val = u64(val)
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (low_u32, high_u32) into a 64-bit qword."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def calc_lds_src2_byte_address(vaddr: int, offset8: int) -> int:
    """Calculate the 32-bit effective byte address for a src2 operand.
    
    offset8 is an 8-bit unsigned immediate multiplied by 8 bytes (qword stride).
    Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + ((offset8 & OFFSET8_MASK) * 8)) & MASK32


def calc_lds_qword_index(byte_addr: int, capacity_qwords: int = DEFAULT_LDS_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS qword index.
    
    Qword addresses are 8-byte aligned (low 3 bits dropped).
    """
    return (byte_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank_pair(byte_addr: int) -> Tuple[int, int]:
    """Determine the two LDS memory banks (0..31) occupied by a 64-bit qword."""
    dword_idx = (byte_addr >> 2) & 31
    return dword_idx, (dword_idx + 1) & 31


def oracle_ds_add_src2_u64(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_add_src2_u64: (val_a + val_b) mod 2^64."""
    return u64(u64(val_a) + u64(val_b))


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_qwords: int = DEFAULT_LDS_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory = [0] * capacity_qwords

    def read_qword(self, qword_idx: int) -> int:
        idx = qword_idx & (self.capacity_qwords - 1)
        return self.memory[idx]

    def write_qword(self, qword_idx: int, val: int) -> None:
        idx = qword_idx & (self.capacity_qwords - 1)
        self.memory[idx] = u64(val)

    def read_qword_at_byte_addr(self, byte_addr: int) -> int:
        idx = calc_lds_qword_index(byte_addr, self.capacity_qwords)
        return self.memory[idx]

    def write_qword_at_byte_addr(self, byte_addr: int, val: int) -> None:
        idx = calc_lds_qword_index(byte_addr, self.capacity_qwords)
        self.memory[idx] = u64(val)

    def atomic_add_src2_u64(self, vaddr: int, offset0: int, offset1: int) -> int:
        """Execute atomic dual-source 64-bit addition on LDS.
        
        LDS[A] = (LDS[A] + LDS[B]) mod 2^64.
        Returns the updated value at location A.
        """
        byte_addr_a = calc_lds_src2_byte_address(vaddr, offset0)
        byte_addr_b = calc_lds_src2_byte_address(vaddr, offset1)

        idx_a = calc_lds_qword_index(byte_addr_a, self.capacity_qwords)
        idx_b = calc_lds_qword_index(byte_addr_b, self.capacity_qwords)

        val_a = self.memory[idx_a]
        val_b = self.memory[idx_b]

        new_val_a = oracle_ds_add_src2_u64(val_a, val_b)
        self.memory[idx_a] = new_val_a
        return new_val_a


class TestLdsAddSrc2U64ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_basic_add_src2(self) -> None:
        """LDS[A] += LDS[B] with distinct locations."""
        vaddr = 0x100
        offset0 = 2  # A = 0x100 + 16 = 0x110
        offset1 = 5  # B = 0x100 + 40 = 0x128

        self.lds.write_qword_at_byte_addr(0x110, 1000)
        self.lds.write_qword_at_byte_addr(0x128, 250)

        res = self.lds.atomic_add_src2_u64(vaddr, offset0, offset1)
        self.assertEqual(res, 1250)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x110), 1250)
        # Location B must remain unchanged
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x128), 250)

    def test_identity_zero(self) -> None:
        """When LDS[B] is zero, LDS[A] remains unchanged."""
        vaddr = 0x200
        self.lds.write_qword_at_byte_addr(0x200, 0x1234_5678_9ABC_DEF0)
        self.lds.write_qword_at_byte_addr(0x208, 0)

        res = self.lds.atomic_add_src2_u64(vaddr, 0, 1)
        self.assertEqual(res, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x200), 0x1234_5678_9ABC_DEF0)

    def test_aliased_addresses_same_offset(self) -> None:
        """When offset0 == offset1, A == B: LDS[A] becomes 2 * LDS[A]."""
        vaddr = 0x300
        val = 0x0000_0001_0000_0000
        self.lds.write_qword_at_byte_addr(0x300, val)

        res = self.lds.atomic_add_src2_u64(vaddr, 0, 0)
        self.assertEqual(res, 0x0000_0002_0000_0000)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x300), 0x0000_0002_0000_0000)

    def test_64bit_overflow_wrap(self) -> None:
        """64-bit unsigned addition overflow wraps modulo 2^64."""
        vaddr = 0x400
        self.lds.write_qword_at_byte_addr(0x400, MASK64)
        self.lds.write_qword_at_byte_addr(0x408, 1)

        res = self.lds.atomic_add_src2_u64(vaddr, 0, 1)
        self.assertEqual(res, 0)
        self.assertEqual(self.lds.read_qword_at_byte_addr(0x400), 0)

    def test_stride_8_offset_scaling(self) -> None:
        """offset8 is multiplied by 8 bytes (qword stride)."""
        base = 0x1000
        addr_off0 = calc_lds_src2_byte_address(base, 0)
        addr_off1 = calc_lds_src2_byte_address(base, 1)
        addr_off2 = calc_lds_src2_byte_address(base, 2)
        addr_off255 = calc_lds_src2_byte_address(base, 255)

        self.assertEqual(addr_off0, 0x1000)
        self.assertEqual(addr_off1, 0x1008)
        self.assertEqual(addr_off2, 0x1010)
        self.assertEqual(addr_off255, 0x1000 + 255 * 8)

    def test_qword_alignment_truncation(self) -> None:
        """Low 3 bits of byte address are ignored for qword alignment."""
        idx0 = calc_lds_qword_index(0x800)
        idx1 = calc_lds_qword_index(0x801)
        idx7 = calc_lds_qword_index(0x807)
        self.assertEqual(idx0, 0x100)
        self.assertEqual(idx1, 0x100)
        self.assertEqual(idx7, 0x100)

    def test_bank_pair_distribution(self) -> None:
        """Each qword occupies a pair of consecutive 32-bit LDS memory banks."""
        b_lo, b_hi = calc_lds_bank_pair(0)
        self.assertEqual((b_lo, b_hi), (0, 1))

        b_lo, b_hi = calc_lds_bank_pair(31 * 4)  # Last dword bank 31 wraps to 0
        self.assertEqual((b_lo, b_hi), (31, 0))

    def test_capacity_wrapping(self) -> None:
        """Qword access wraps modulo capacity (default 64 KiB = 8192 qwords)."""
        addr_base = 0x10
        addr_wrapped = addr_base + (DEFAULT_LDS_QWORDS * 8)
        idx_base = calc_lds_qword_index(addr_base)
        idx_wrapped = calc_lds_qword_index(addr_wrapped)
        self.assertEqual(idx_base, idx_wrapped)


if __name__ == "__main__":
    unittest.main()


# ============================================================================
# Section 2: Symbolic AST Opcode Definitions and Typed Expressions
# ============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_VGPRS):
            raise ValueError(f"VGPR index out of range [0, {DEFAULT_NUM_VGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_SGPRS):
            raise ValueError(f"SGPR index out of range [0, {DEFAULT_NUM_SGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclass(frozen=True)
class DSAddSrc2U64Instruction:
    vaddr: VGPR
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.offset0 <= OFFSET8_MASK):
            raise ValueError(f"offset0 out of 8-bit range [0, 255]: {self.offset0}")
        if not (0 <= self.offset1 <= OFFSET8_MASK):
            raise ValueError(f"offset1 out of 8-bit range [0, 255]: {self.offset1}")
        if not isinstance(self.vaddr, VGPR):
            raise TypeError(f"vaddr must be a VGPR, got {type(self.vaddr)}")

    def disasm(self) -> str:
        tokens = ["ds_add_src2_u64", str(self.vaddr)]
        if self.offset0 != 0:
            tokens.append(f"offset0:{self.offset0}")
        if self.offset1 != 0:
            tokens.append(f"offset1:{self.offset1}")
        if self.gds:
            tokens.append("gds")
        return " ".join(tokens)


class ASTExpr64:
    """Base class for symbolic typed expressions in RDNA2 64-bit dual-source arithmetic."""

    def simplify(self) -> ASTExpr64:
        raise NotImplementedError

    def evaluate(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Symbol64(ASTExpr64):
    name: str

    def simplify(self) -> ASTExpr64:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Symbol '{self.name}' not found in evaluation environment")
        return u64(env[self.name])

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant64(ASTExpr64):
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u64(self.value))

    def simplify(self) -> ASTExpr64:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        return self.value

    def __str__(self) -> str:
        return f"0x{self.value:016X}"


@dataclass(frozen=True)
class DualSourceAddExpr(ASTExpr64):
    loc_a: ASTExpr64
    loc_b: ASTExpr64

    def simplify(self) -> ASTExpr64:
        s_a = self.loc_a.simplify()
        s_b = self.loc_b.simplify()

        # Constant folding
        if isinstance(s_a, Constant64) and isinstance(s_b, Constant64):
            return Constant64(s_a.value + s_b.value)

        # Identity addition: A + 0 -> A
        if isinstance(s_b, Constant64) and s_b.value == 0:
            return s_a

        return DualSourceAddExpr(s_a, s_b)

    def evaluate(self, env: Dict[str, int]) -> int:
        return u64(self.loc_a.evaluate(env) + self.loc_b.evaluate(env))

    def __str__(self) -> str:
        return f"({self.loc_a} + {self.loc_b})"


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_valid_creation_and_disassembly(self) -> None:
        instr = DSAddSrc2U64Instruction(vaddr=VGPR(4), offset0=2, offset1=6)
        self.assertEqual(instr.disasm(), "ds_add_src2_u64 v4 offset0:2 offset1:6")

        instr_default = DSAddSrc2U64Instruction(vaddr=VGPR(1), offset0=0, offset1=0, gds=True)
        self.assertEqual(instr_default.disasm(), "ds_add_src2_u64 v1 gds")

    def test_instruction_validation_vgpr_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)

    def test_instruction_validation_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=256, offset1=0)
        with self.assertRaises(ValueError):
            DSAddSrc2U64Instruction(vaddr=VGPR(0), offset0=0, offset1=-1)

    def test_symbolic_constant_folding(self) -> None:
        expr = DualSourceAddExpr(Constant64(500), Constant64(250))
        simplified = expr.simplify()
        self.assertIsInstance(simplified, Constant64)
        self.assertEqual(simplified.evaluate({}), 750)

    def test_symbolic_identity_zero(self) -> None:
        a = Symbol64("LDS_A")
        expr = DualSourceAddExpr(a, Constant64(0))
        simplified = expr.simplify()
        self.assertEqual(simplified, a)

    def test_symbolic_evaluation_env(self) -> None:
        a = Symbol64("A")
        b = Symbol64("B")
        expr = DualSourceAddExpr(a, b)
        env = {"A": 1000, "B": 2000}
        self.assertEqual(expr.evaluate(env), 3000)
