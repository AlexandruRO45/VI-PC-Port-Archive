from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_inc_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Increment with Return:
Performs a 64-bit atomic wrapping increment on LDS memory and returns the original 64-bit value.

Syntax:
    ds_inc_rtn_u64 vdst, vaddr, vdata0 [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS[qword_index]
    new_mem = 0 if old_val >= (vdata0 & 0xFFFFFFFFFFFFFFFF) else (old_val + 1) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_mem
    vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Capacity: 64 KiB default capacity (8,192 64-bit qwords / 16,384 32-bit dwords).
  - Addressing: Base byte address in VGPR vaddr + 16-bit unsigned immediate offset.
  - Alignment: 64-bit qword aligned; low 3 bits of byte address are truncated (eff_addr >> 3).
  - Wrap Condition: Unsigned 64-bit comparison: if old_val >= wrap_limit then new_mem = 0,
    else new_mem = (old_val + 1) mod 2^64.
  - Return: The pre-operation 64-bit value is returned to VGPR pair vdst.
"""

import struct
import unittest
from typing import Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536          # 64 KiB capacity
LDS_SIZE_DWORDS: int = 16384         # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192          # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32             # 32 memory banks
MASK_32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF          # 16-bit offset mask
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
DEFAULT_NUM_LANES: int = 64


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK_64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as signed two's-complement integer."""
    val &= MASK_64
    return val - (1 << 64) if (val & (1 << 63)) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index."""
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the starting LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_inc_rtn_u64(
    current_mem: int,
    wrap_limit: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_inc_rtn_u64.

    Computes:
        old_val = current_mem & 0xFFFFFFFFFFFFFFFF
        limit = wrap_limit & 0xFFFFFFFFFFFFFFFF
        new_mem = 0 if old_val >= limit else (old_val + 1) & 0xFFFFFFFFFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u64(current_mem)
    limit = u64(wrap_limit)
    new_mem = 0 if old_val >= limit else u64(old_val + 1)
    return old_val, new_mem


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad with 64-bit access."""

    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a positive power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory = [0] * capacity_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        return self.memory[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        self.memory[idx] = u64(val)

    def atomic_inc_rtn_u64(
        self,
        vaddr: int,
        wrap_limit: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit wrapping increment with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_inc_rtn_u64(curr, wrap_limit)
        self.memory[idx] = new_mem
        return old_val, new_mem


class TestReferenceOracleAndLDSModel64(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model for ds_inc_rtn_u64."""

    def test_oracle_normal_increment_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(0, 5)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 1)

        old, new_mem = oracle_ds_inc_rtn_u64(4, 5)
        self.assertEqual(old, 4)
        self.assertEqual(new_mem, 5)

        old, new_mem = oracle_ds_inc_rtn_u64(0x1_0000_0000, 0x2_0000_0000)
        self.assertEqual(old, 0x1_0000_0000)
        self.assertEqual(new_mem, 0x1_0000_0001)

    def test_oracle_wrap_on_equal_threshold_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(5, 5)
        self.assertEqual(old, 5)
        self.assertEqual(new_mem, 0)

        limit_64 = 0x1234_5678_9ABC_DEF0
        old, new_mem = oracle_ds_inc_rtn_u64(limit_64, limit_64)
        self.assertEqual(old, limit_64)
        self.assertEqual(new_mem, 0)

    def test_oracle_wrap_on_greater_threshold_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(6, 5)
        self.assertEqual(old, 6)
        self.assertEqual(new_mem, 0)

        old, new_mem = oracle_ds_inc_rtn_u64(0xFFFF_FFFF_FFFF_FFFF, 100)
        self.assertEqual(old, 0xFFFF_FFFF_FFFF_FFFF)
        self.assertEqual(new_mem, 0)

    def test_oracle_zero_wrap_limit_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(0, 0)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 0)

        old, new_mem = oracle_ds_inc_rtn_u64(42, 0)
        self.assertEqual(old, 42)
        self.assertEqual(new_mem, 0)

    def test_oracle_max_u64_limit(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(0, MASK_64)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 1)

        old, new_mem = oracle_ds_inc_rtn_u64(MASK_64 - 1, MASK_64)
        self.assertEqual(old, MASK_64 - 1)
        self.assertEqual(new_mem, MASK_64)

        old, new_mem = oracle_ds_inc_rtn_u64(MASK_64, MASK_64)
        self.assertEqual(old, MASK_64)
        self.assertEqual(new_mem, 0)

    def test_effective_address_and_offset_u64(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)

        # 32-bit modulo wrapping
        eff_wrap = calc_lds_effective_address(0xFFFF_FFF8, 0x0010)
        self.assertEqual(eff_wrap, 0x0008)

        # 16-bit offset mask
        eff_mask = calc_lds_effective_address(0x2000, 0x1_0020)
        self.assertEqual(eff_mask, 0x2020)

    def test_qword_alignment_truncation(self) -> None:
        # Low 3 bits truncated (64-bit qword alignment)
        for offset in range(8):
            self.assertEqual(calc_lds_qword_index(0x1000 + offset), 0x200)
        self.assertEqual(calc_lds_qword_index(0x1008), 0x201)

    def test_qword_capacity_wrapping(self) -> None:
        capacity_bytes = LDS_SIZE_QWORDS * 8
        eff = calc_lds_effective_address(capacity_bytes + 0x80)
        idx = calc_lds_qword_index(eff, LDS_SIZE_QWORDS)
        self.assertEqual(idx, 0x10)

    def test_lds_model_atomic_inc_rtn_u64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        addr = 0x80

        lds.write_qword(addr, 0x1_0000_0000)
        old, new_mem = lds.atomic_inc_rtn_u64(vaddr=addr, wrap_limit=0x2_0000_0000, offset=0)
        self.assertEqual(old, 0x1_0000_0000)
        self.assertEqual(new_mem, 0x1_0000_0001)
        self.assertEqual(lds.read_qword(addr), 0x1_0000_0001)

        # Increment with offset
        old, new_mem = lds.atomic_inc_rtn_u64(vaddr=0x78, wrap_limit=0x2_0000_0000, offset=8)
        self.assertEqual(old, 0x1_0000_0001)
        self.assertEqual(new_mem, 0x1_0000_0002)
        self.assertEqual(lds.read_qword(addr), 0x1_0000_0002)

        # Hit wrap condition
        old, new_mem = lds.atomic_inc_rtn_u64(vaddr=addr, wrap_limit=0x1_0000_0002, offset=0)
        self.assertEqual(old, 0x1_0000_0002)
        self.assertEqual(new_mem, 0)
        self.assertEqual(lds.read_qword(addr), 0)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

import dataclasses
import enum


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
        if self.index < 0 or self.index >= DEFAULT_NUM_VGPRS:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2, 4):
            raise ValueError(f"Unsupported register count: {self.count}")
        if self.index + self.count > DEFAULT_NUM_VGPRS:
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


class Ds64Opcode(enum.Enum):
    """LDS 64-bit instruction opcodes."""
    DS_INC_RTN_U64 = "ds_inc_rtn_u64"
    DS_INC_U64 = "ds_inc_u64"


@dataclasses.dataclass(frozen=True)
class Ds64Instruction:
    """Symbolic AST node representing a 64-bit DS atomic increment instruction."""
    opcode: Ds64Opcode
    vaddr: Register
    vdata0: Register
    vdst: Register | None = None
    offset: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR: {self.vaddr}")
        if not self.vdata0.is_vgpr or self.vdata0.count != 2:
            raise ValueError(f"vdata0 must be a 64-bit VGPR pair (count=2): {self.vdata0}")
        if self.opcode == Ds64Opcode.DS_INC_RTN_U64:
            if self.vdst is None or not self.vdst.is_vgpr or self.vdst.count != 2:
                raise ValueError(f"ds_inc_rtn_u64 requires a 64-bit destination VGPR pair vdst (count=2): {self.vdst}")
        elif self.opcode == Ds64Opcode.DS_INC_U64:
            if self.vdst is not None:
                raise ValueError(f"ds_inc_u64 does not accept a destination register: {self.vdst}")
        if self.offset < 0 or self.offset > 65535:
            raise ValueError(f"LDS offset must be 16-bit unsigned [0, 65535], got {self.offset}")

    def to_assembly(self) -> str:
        parts: list[str] = [self.opcode.value]
        if self.vdst is not None:
            parts.append(f"{self.vdst.format_asm()},")
        parts.append(f"{self.vaddr.format_asm()},")
        parts.append(self.vdata0.format_asm())
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")

        first_arg = parts[1]
        rest = " ".join(parts[2:])
        return f"{self.opcode.value} {first_arg} {rest}" if rest else f"{self.opcode.value} {first_arg}"


class Ds64IncExpr:
    """Typed symbolic expression for ds_inc_rtn_u64."""

    def __init__(self, vaddr: int, wrap_limit: int, offset: int = 0) -> None:
        self.vaddr = vaddr & MASK_32
        self.wrap_limit = wrap_limit & MASK_64
        self.offset = offset & OFFSET16_MASK

    def evaluate(self, lds: LDSModel64) -> Tuple[int, int]:
        return lds.atomic_inc_rtn_u64(self.vaddr, self.wrap_limit, self.offset)


class TestDsIncRtnU64AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_values(self) -> None:
        self.assertEqual(Ds64Opcode.DS_INC_RTN_U64.value, "ds_inc_rtn_u64")
        self.assertEqual(Ds64Opcode.DS_INC_U64.value, "ds_inc_u64")

    def test_register_operand_formatting(self) -> None:
        v1 = Register(RegKind.VGPR, 1, count=1)
        self.assertEqual(v1.format_asm(), "v1")
        v_pair = Register(RegKind.VGPR, 4, count=2)
        self.assertEqual(v_pair.format_asm(), "v[4:5]")

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 255, count=2)

    def test_instruction_ast_returning(self) -> None:
        vdst = Register(RegKind.VGPR, 0, count=2)
        vaddr = Register(RegKind.VGPR, 2, count=1)
        vdata = Register(RegKind.VGPR, 4, count=2)

        instr = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=vdst,
            vaddr=vaddr,
            vdata0=vdata,
            offset=16,
        )
        self.assertEqual(instr.to_assembly(), "ds_inc_rtn_u64 v[0:1], v2, v[4:5] offset:16")

    def test_instruction_ast_non_returning(self) -> None:
        vaddr = Register(RegKind.VGPR, 2, count=1)
        vdata = Register(RegKind.VGPR, 4, count=2)

        instr = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_U64,
            vaddr=vaddr,
            vdata0=vdata,
            offset=0,
        )
        self.assertEqual(instr.to_assembly(), "ds_inc_u64 v2, v[4:5]")

    def test_instruction_validation_errors(self) -> None:
        v1 = Register(RegKind.VGPR, 1, count=1)
        v2 = Register(RegKind.VGPR, 2, count=2)
        s1 = Register(RegKind.SGPR, 1, count=1)

        # vaddr must be VGPR
        with self.assertRaises(ValueError):
            Ds64Instruction(opcode=Ds64Opcode.DS_INC_U64, vaddr=s1, vdata0=v2)
        # vdata0 must be count=2
        with self.assertRaises(ValueError):
            Ds64Instruction(opcode=Ds64Opcode.DS_INC_U64, vaddr=v1, vdata0=v1)
        # returning requires vdst count=2
        with self.assertRaises(ValueError):
            Ds64Instruction(opcode=Ds64Opcode.DS_INC_RTN_U64, vaddr=v1, vdata0=v2, vdst=v1)
        # non-returning must not have vdst
        with self.assertRaises(ValueError):
            Ds64Instruction(opcode=Ds64Opcode.DS_INC_U64, vaddr=v1, vdata0=v2, vdst=v2)
        # offset out of range
        with self.assertRaises(ValueError):
            Ds64Instruction(opcode=Ds64Opcode.DS_INC_U64, vaddr=v1, vdata0=v2, offset=70000)

    def test_typed_expression_evaluation(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        addr = 0x100
        lds.write_qword(addr, 10)

        expr = Ds64IncExpr(vaddr=addr, wrap_limit=15, offset=0)
        old, new_mem = expr.evaluate(lds)
        self.assertEqual(old, 10)
        self.assertEqual(new_mem, 11)
        self.assertEqual(lds.read_qword(addr), 11)


# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Scalar register file maintaining 256 VGPRs and 106 SGPRs."""

    def __init__(self) -> None:
        self.vgprs = [0] * DEFAULT_NUM_VGPRS
        self.sgprs = [0] * DEFAULT_NUM_SGPRS

    def read_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK_32

    def read_vgpr_64(self, idx: int) -> int:
        lo = self.vgprs[idx] & MASK_32
        hi = self.vgprs[idx + 1] & MASK_32
        return lo | (hi << 32)

    def write_vgpr_64(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK_32
        self.vgprs[idx + 1] = (val >> 32) & MASK_32

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK_32


class ScalarLDS64Interpreter:
    """Deterministic scalar interpreter executing 64-bit LDS atomic instructions."""

    def __init__(self, lds: LDSModel64, rf: ScalarRegisterFile | None = None) -> None:
        self.lds = lds
        self.rf = rf if rf is not None else ScalarRegisterFile()

    def step(self, inst: Ds64Instruction) -> Tuple[int, int]:
        vaddr = self.rf.read_vgpr(inst.vaddr.index)
        wrap_limit = self.rf.read_vgpr_64(inst.vdata0.index)

        old_val, new_mem = self.lds.atomic_inc_rtn_u64(
            vaddr=vaddr,
            wrap_limit=wrap_limit,
            offset=inst.offset,
        )

        if inst.opcode == Ds64Opcode.DS_INC_RTN_U64 and inst.vdst is not None:
            self.rf.write_vgpr_64(inst.vdst.index, old_val)

        return old_val, new_mem


class TestDsIncRtnU64ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSModel64(capacity_qwords=512)
        self.rf = ScalarRegisterFile()
        self.interp = ScalarLDS64Interpreter(self.lds, self.rf)

    def test_scalar_step_normal_increment_u64(self) -> None:
        addr = 0x100
        self.lds.write_qword(addr, 0x1_0000_0000)
        self.rf.write_vgpr(2, addr)
        self.rf.write_vgpr_64(4, 0x2_0000_0000)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
            offset=0,
        )

        old, new_mem = self.interp.step(inst)
        self.assertEqual(old, 0x1_0000_0000)
        self.assertEqual(new_mem, 0x1_0000_0001)
        self.assertEqual(self.rf.read_vgpr_64(0), 0x1_0000_0000)
        self.assertEqual(self.lds.read_qword(addr), 0x1_0000_0001)

    def test_scalar_step_wrap_equal_threshold_u64(self) -> None:
        addr = 0x140
        self.lds.write_qword(addr, 50)
        self.rf.write_vgpr(2, addr)
        self.rf.write_vgpr_64(4, 50)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        old, new_mem = self.interp.step(inst)
        self.assertEqual(old, 50)
        self.assertEqual(new_mem, 0)
        self.assertEqual(self.rf.read_vgpr_64(0), 50)
        self.assertEqual(self.lds.read_qword(addr), 0)

    def test_scalar_step_wrap_greater_threshold_u64(self) -> None:
        addr = 0x180
        self.lds.write_qword(addr, 100)
        self.rf.write_vgpr(2, addr)
        self.rf.write_vgpr_64(4, 50)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        old, new_mem = self.interp.step(inst)
        self.assertEqual(old, 100)
        self.assertEqual(new_mem, 0)
        self.assertEqual(self.rf.read_vgpr_64(0), 100)

    def test_scalar_step_with_offset_u64(self) -> None:
        base = 0x200
        offset = 0x18  # 24 bytes = 3 qwords
        target = base + offset

        self.lds.write_qword(target, 42)
        self.rf.write_vgpr(2, base)
        self.rf.write_vgpr_64(4, 1000)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
            offset=offset,
        )

        old, new_mem = self.interp.step(inst)
        self.assertEqual(old, 42)
        self.assertEqual(new_mem, 43)
        self.assertEqual(self.lds.read_qword(target), 43)

    def test_scalar_step_non_returning_does_not_modify_vdst(self) -> None:
        addr = 0x80
        self.lds.write_qword(addr, 5)
        self.rf.write_vgpr(2, addr)
        self.rf.write_vgpr_64(4, 10)
        # Pre-seed v[0:1]
        self.rf.write_vgpr_64(0, 0xDEAD_BEEF_CAFE_BABE)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_U64,
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        old, new_mem = self.interp.step(inst)
        self.assertEqual(old, 5)
        self.assertEqual(new_mem, 6)
        self.assertEqual(self.lds.read_qword(addr), 6)
        self.assertEqual(self.rf.read_vgpr_64(0), 0xDEAD_BEEF_CAFE_BABE)

    def test_scalar_step_destination_writeback_both_dwords(self) -> None:
        addr = 0xC0
        val64 = 0x1234_5678_9ABC_DEF0
        self.lds.write_qword(addr, val64)
        self.rf.write_vgpr(2, addr)
        self.rf.write_vgpr_64(4, MASK_64)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 10, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        self.interp.step(inst)
        self.assertEqual(self.rf.read_vgpr(10), 0x9ABC_DEF0)
        self.assertEqual(self.rf.read_vgpr(11), 0x1234_5678)
        self.assertEqual(self.rf.read_vgpr_64(10), val64)


if __name__ == "__main__":
    unittest.main()
