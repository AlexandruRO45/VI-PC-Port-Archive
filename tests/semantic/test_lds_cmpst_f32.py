from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_cmpst_f32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Floating-Point Compare-and-Swap:
Performs an atomic IEEE 754 single-precision float comparison on LDS memory and swaps if equal.

Syntax:
    ds_cmpst_f32 vaddr, vdata0, vdata1 [offset:offset16] [gds]
    ds_cmpst_rtn_f32 vdst, vaddr, vdata0, vdata1 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_raw = LDS[dword_index]
    if float_equal_f32(old_raw, vdata0):
        LDS[dword_index] = vdata1
    if returning:
        vdst = old_raw
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
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
MIN_NORMAL_F32: int = 0x00800000
MIN_SUBNORMAL_F32: int = 0x00000001
MAX_SUBNORMAL_F32: int = 0x007FFFFF
MAX_NORMAL_F32: int = 0x7F7FFFFF
ONE_F32: int = 0x3F800000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 subnormal float."""
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


def float_equal_f32(a_bits: int, b_bits: int) -> bool:
    """IEEE 754 single-precision equality comparison.

    NaNs compare unordered (always False).
    Signed zeros (+0.0 and -0.0) compare equal.
    """
    if is_f32_nan(a_bits) or is_f32_nan(b_bits):
        return False
    return bits_to_f32(a_bits) == bits_to_f32(b_bits)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_cmpst_f32(
    current_mem: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_cmpst_f32.

    Compares current_mem with cmp_val using IEEE 754 float equality;
    if equal, replaces memory with new_val.
    Always returns (old_mem, new_mem, swapped).
    """
    old_raw = u32(current_mem)
    cmp_raw = u32(cmp_val)
    new_raw = u32(new_val)

    if float_equal_f32(old_raw, cmp_raw):
        return old_raw, new_raw, True
    return old_raw, old_raw, False


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.memory[idx] = u32(val)

    def atomic_cmpst_f32(
        self,
        vaddr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        curr = self.memory[idx]
        old_val, new_mem, swapped = ref_ds_cmpst_f32(curr, cmp_val, new_val)
        self.memory[idx] = new_mem
        return old_val, new_mem, swapped



# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    DS_CMPST_F32 = "ds_cmpst_f32"
    DS_CMPST_RTN_F32 = "ds_cmpst_rtn_f32"


class RegType(Enum):
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(f"Register index out of bounds: {self.reg_type.value}[{self.index}]")

    def __str__(self) -> str:
        return f"{self.reg_type.value}[{self.index}]"


@dataclass(frozen=True)
class ImmediateF32:
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u32(self.val))

    def __str__(self) -> str:
        return f"0x{self.val:X}"


Operand = Union[RegisterOperand, ImmediateF32]


@dataclass(frozen=True)
class DSCmpstF32Instruction:
    """Symbolic AST node for ds_cmpst_f32 / ds_cmpst_rtn_f32 instruction."""
    vaddr: RegisterOperand
    vdata0: Operand  # Float compare value
    vdata1: Operand  # Float new value
    vdst: Optional[RegisterOperand] = None  # None for ds_cmpst_f32, present for ds_cmpst_rtn_f32
    offset: int = 0
    gds: bool = False

    @property
    def opcode(self) -> DSOpcode:
        return DSOpcode.DS_CMPST_RTN_F32 if self.vdst is not None else DSOpcode.DS_CMPST_F32

    def __post_init__(self) -> None:
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR, got {self.vaddr.reg_type}")
        if self.vdst is not None and self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be a VGPR, got {self.vdst.reg_type}")
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



# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarState:
    """Scalar machine state comprising register files and LDS memory."""

    def __init__(self, lds_size_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_size_dwords)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)

    def get_reg(self, reg: RegisterOperand) -> int:
        return self.get_vgpr(reg.index) if reg.reg_type == RegType.VGPR else self.get_sgpr(reg.index)

    def set_reg(self, reg: RegisterOperand, val: int) -> None:
        if reg.reg_type == RegType.VGPR:
            self.set_vgpr(reg.index, val)
        else:
            self.set_sgpr(reg.index, val)

    def resolve_operand(self, op: Operand) -> int:
        return op.val if isinstance(op, ImmediateF32) else self.get_reg(op)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_cmpst_f32."""

    def execute(self, instr: DSCmpstF32Instruction, state: ScalarState) -> Tuple[int, int, bool]:
        vaddr = state.get_reg(instr.vaddr)
        cmp_val = state.resolve_operand(instr.vdata0)
        new_val = state.resolve_operand(instr.vdata1)

        old_val, new_mem, swapped = state.lds.atomic_cmpst_f32(
            vaddr=vaddr,
            cmp_val=cmp_val,
            new_val=new_val,
            offset=instr.offset,
        )

        if instr.vdst is not None:
            state.set_reg(instr.vdst, old_val)

        return old_val, new_mem, swapped



class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_finite_float_match(self) -> None:
        val = f32_to_bits(3.14159)
        new_val = f32_to_bits(2.71828)
        old, mem, swapped = ref_ds_cmpst_f32(val, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)

    def test_oracle_finite_float_mismatch(self) -> None:
        val = f32_to_bits(10.0)
        cmp_val = f32_to_bits(20.0)
        old, mem, swapped = ref_ds_cmpst_f32(val, cmp_val, f32_to_bits(50.0))
        self.assertFalse(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, val)

    def test_oracle_signed_zero_equality(self) -> None:
        # In IEEE 754, +0.0 == -0.0 is True!
        old, mem, swapped = ref_ds_cmpst_f32(NEG_ZERO_F32, POS_ZERO_F32, f32_to_bits(1.0))
        self.assertTrue(swapped)
        self.assertEqual(old, NEG_ZERO_F32)
        self.assertEqual(mem, f32_to_bits(1.0))

        old2, mem2, swapped2 = ref_ds_cmpst_f32(POS_ZERO_F32, NEG_ZERO_F32, f32_to_bits(2.0))
        self.assertTrue(swapped2)
        self.assertEqual(old2, POS_ZERO_F32)
        self.assertEqual(mem2, f32_to_bits(2.0))

    def test_oracle_nan_unordered_rejection(self) -> None:
        # NaN == NaN evaluates to False
        old, mem, swapped = ref_ds_cmpst_f32(QNAN_F32, QNAN_F32, f32_to_bits(1.0))
        self.assertFalse(swapped)
        self.assertEqual(mem, QNAN_F32)

    def test_oracle_infinities(self) -> None:
        # +inf == +inf is True, -inf == -inf is True, +inf == -inf is False
        old, mem, swapped = ref_ds_cmpst_f32(POS_INF_F32, POS_INF_F32, 0)
        self.assertTrue(swapped)
        self.assertEqual(mem, 0)

        old2, mem2, swapped2 = ref_ds_cmpst_f32(POS_INF_F32, NEG_INF_F32, 0)
        self.assertFalse(swapped2)
        self.assertEqual(mem2, POS_INF_F32)

    def test_effective_address_and_alignment(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x14)
        self.assertEqual(eff, 0x1014)
        self.assertEqual(calc_lds_dword_index(eff), 0x1014 >> 2)

    def test_architectural_memory_atomic_cmpst_f32(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x80
        val = f32_to_bits(42.0)
        new_val = f32_to_bits(84.0)

        lds.write_dword(addr, val)
        old, mem, swapped = lds.atomic_cmpst_f32(addr, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_dword(addr), new_val)



class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_non_returning_ast(self) -> None:
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)
        vdata1 = RegisterOperand(3, RegType.VGPR)
        instr = DSCmpstF32Instruction(vaddr=vaddr, vdata0=vdata0, vdata1=vdata1, offset=16)

        self.assertEqual(instr.opcode, DSOpcode.DS_CMPST_F32)
        self.assertIsNone(instr.vdst)
        self.assertEqual(instr.disassemble(), "ds_cmpst_f32 v[1], v[2], v[3], offset:16")

    def test_returning_ast(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = ImmediateF32(f32_to_bits(1.0))
        vdata1 = ImmediateF32(f32_to_bits(2.0))
        instr = DSCmpstF32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, vdata1=vdata1)

        self.assertEqual(instr.opcode, DSOpcode.DS_CMPST_RTN_F32)
        self.assertEqual(instr.disassemble(), "ds_cmpst_rtn_f32 v[0], v[1], 0x3F800000, 0x40000000")

    def test_ast_validation_errors(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        with self.assertRaises(ValueError):
            DSCmpstF32Instruction(vaddr=s0, vdata0=v0, vdata1=v0)
        with self.assertRaises(ValueError):
            DSCmpstF32Instruction(vdst=s0, vaddr=v0, vdata0=v0, vdata1=v0)
        with self.assertRaises(ValueError):
            DSCmpstF32Instruction(vaddr=v0, vdata0=v0, vdata1=v0, offset=-1)
        with self.assertRaises(ValueError):
            DSCmpstF32Instruction(vaddr=v0, vdata0=v0, vdata1=v0, offset=65536)



class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarState(lds_size_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_non_returning_cmpst_f32(self) -> None:
        addr = 0x40
        val = f32_to_bits(12.5)
        new_val = f32_to_bits(25.0)

        self.state.lds.write_dword(addr, val)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, val)
        self.state.set_vgpr(3, new_val)

        instr = DSCmpstF32Instruction(
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)
        self.assertTrue(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, new_val)
        self.assertEqual(self.state.lds.read_dword(addr), new_val)

    def test_scalar_returning_cmpst_rtn_f32(self) -> None:
        addr = 0x40
        val = f32_to_bits(100.0)
        new_val = f32_to_bits(200.0)

        self.state.lds.write_dword(addr, val)
        self.state.set_vgpr(0, 0xCAFE) # vdst sentinel
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, val)
        self.state.set_vgpr(3, new_val)

        instr = DSCmpstF32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)
        self.assertTrue(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(self.state.get_vgpr(0), val)
        self.assertEqual(self.state.lds.read_dword(addr), new_val)

    def test_scalar_mismatch_no_write(self) -> None:
        addr = 0x40
        val = f32_to_bits(1.0)
        self.state.lds.write_dword(addr, val)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, f32_to_bits(2.0)) # Mismatch
        self.state.set_vgpr(3, f32_to_bits(99.0))

        instr = DSCmpstF32Instruction(
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            vdata1=RegisterOperand(3, RegType.VGPR),
        )

        old_val, new_mem, swapped = self.interp.execute(instr, self.state)
        self.assertFalse(swapped)
        self.assertEqual(self.state.lds.read_dword(addr), val)



if __name__ == "__main__":
    unittest.main()
