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


# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDLDS64WavefrontInterpreter:
    """Multi-lane SIMD wavefront interpreter for ds_inc_rtn_u64 with EXEC masking."""

    def __init__(self, wave_size: int = 64, lds: LDSModel64 | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.lds = lds if lds is not None else LDSModel64()
        self.vgprs = [[0] * DEFAULT_NUM_VGPRS for _ in range(wave_size)]

    @property
    def exec_mask(self) -> int:
        return self._exec_mask

    def set_exec_mask(self, mask: int) -> None:
        limit = (1 << self.wave_size) - 1
        self._exec_mask = mask & limit

    @property
    def exec_lo(self) -> int:
        return self._exec_mask & MASK_32

    def set_exec_lo(self, val: int) -> None:
        self._exec_mask = (self._exec_mask & ~MASK_32) | (val & MASK_32)

    @property
    def exec_hi(self) -> int:
        return (self._exec_mask >> 32) & MASK_32

    def set_exec_hi(self, val: int) -> None:
        if self.wave_size == 32:
            raise ValueError("Cannot set exec_hi on Wave32")
        self._exec_mask = (self._exec_mask & MASK_32) | ((val & MASK_32) << 32)

    def read_lane_vgpr(self, lane: int, idx: int) -> int:
        return self.vgprs[lane][idx]

    def write_lane_vgpr(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = val & MASK_32

    def read_lane_vgpr_64(self, lane: int, idx: int) -> int:
        lo = self.vgprs[lane][idx] & MASK_32
        hi = self.vgprs[lane][idx + 1] & MASK_32
        return lo | (hi << 32)

    def write_lane_vgpr_64(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = val & MASK_32
        self.vgprs[lane][idx + 1] = (val >> 32) & MASK_32

    def step(self, inst: Ds64Instruction) -> list[Tuple[int, int] | None]:
        results: list[Tuple[int, int] | None] = []
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                vaddr = self.read_lane_vgpr(lane, inst.vaddr.index)
                wrap_limit = self.read_lane_vgpr_64(lane, inst.vdata0.index)
                old_val, new_mem = self.lds.atomic_inc_rtn_u64(
                    vaddr=vaddr,
                    wrap_limit=wrap_limit,
                    offset=inst.offset,
                )
                if inst.opcode == Ds64Opcode.DS_INC_RTN_U64 and inst.vdst is not None:
                    self.write_lane_vgpr_64(lane, inst.vdst.index, old_val)
                results.append((old_val, new_mem))
            else:
                results.append(None)
        return results


class TestDsIncRtnU64SIMDMultiLane(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave64_full_exec_all_distinct_lanes(self) -> None:
        lds = LDSModel64(capacity_qwords=1024)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)

        for lane in range(64):
            addr = lane * 8  # Each lane gets its own 64-bit qword
            lds.write_qword(addr, lane * 10)
            simd.write_lane_vgpr(lane, 2, addr)
            simd.write_lane_vgpr_64(lane, 4, 1000)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        results = simd.step(inst)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            addr = lane * 8
            expected_old = lane * 10
            expected_new = lane * 10 + 1
            self.assertEqual(results[lane], (expected_old, expected_new))
            self.assertEqual(simd.read_lane_vgpr_64(lane, 0), expected_old)
            self.assertEqual(lds.read_qword(addr), expected_new)

    def test_wave32_full_exec(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)

        for lane in range(32):
            addr = lane * 8
            lds.write_qword(addr, 100)
            simd.write_lane_vgpr(lane, 2, addr)
            simd.write_lane_vgpr_64(lane, 4, 500)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], (100, 101))
            self.assertEqual(simd.read_lane_vgpr_64(lane, 0), 100)

    def test_exec_mask_disabled_lanes_unmodified(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)
        # Even lanes active
        simd.set_exec_mask(0x5555_5555_5555_5555)

        for lane in range(64):
            addr = lane * 8
            lds.write_qword(addr, 50)
            simd.write_lane_vgpr(lane, 2, addr)
            simd.write_lane_vgpr_64(lane, 4, 200)
            simd.write_lane_vgpr_64(lane, 0, 0xCAFE_BABE)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        results = simd.step(inst)
        for lane in range(64):
            addr = lane * 8
            if lane % 2 == 0:
                self.assertEqual(results[lane], (50, 51))
                self.assertEqual(simd.read_lane_vgpr_64(lane, 0), 50)
                self.assertEqual(lds.read_qword(addr), 51)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(simd.read_lane_vgpr_64(lane, 0), 0xCAFE_BABE)
                self.assertEqual(lds.read_qword(addr), 50)

    def test_exec_lo_hi_partial_wavefront_masking(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)
        simd.set_exec_lo(0x0000_0001)  # Lane 0 only
        simd.set_exec_hi(0x8000_0000)  # Lane 63 only

        self.assertEqual(simd.exec_mask, (1 << 63) | 1)

        for lane in (0, 63):
            addr = lane * 8
            lds.write_qword(addr, 10)
            simd.write_lane_vgpr(lane, 2, addr)
            simd.write_lane_vgpr_64(lane, 4, 50)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        results = simd.step(inst)
        self.assertEqual(results[0], (10, 11))
        self.assertEqual(results[63], (10, 11))
        for lane in range(1, 63):
            self.assertIsNone(results[lane])

    def test_multi_lane_atomic_serialization_same_qword(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        # First 4 lanes active, all targeting same qword
        simd.set_exec_mask(0xF)
        target_addr = 0x80
        lds.write_qword(target_addr, 0)

        for lane in range(4):
            simd.write_lane_vgpr(lane, 2, target_addr)
            simd.write_lane_vgpr_64(lane, 4, 100)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        results = simd.step(inst)
        self.assertEqual(results[0], (0, 1))
        self.assertEqual(results[1], (1, 2))
        self.assertEqual(results[2], (2, 3))
        self.assertEqual(results[3], (3, 4))
        self.assertEqual(lds.read_qword(target_addr), 4)

    def test_multi_lane_serialization_with_wrapping(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        simd.set_exec_mask(0xF)
        target_addr = 0xC0
        lds.write_qword(target_addr, 0)

        for lane in range(4):
            simd.write_lane_vgpr(lane, 2, target_addr)
            simd.write_lane_vgpr_64(lane, 4, 2)  # Wrap limit 2: 0 -> 1 -> 2 -> 0 -> 1

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        results = simd.step(inst)
        self.assertEqual(results[0], (0, 1))
        self.assertEqual(results[1], (1, 2))
        self.assertEqual(results[2], (2, 0))  # Wrapped
        self.assertEqual(results[3], (0, 1))
        self.assertEqual(lds.read_qword(target_addr), 1)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestDsIncRtnU64RegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_vdst_shares_register_with_vaddr(self) -> None:
        # vdst = v[0:1], vaddr = v0 (vaddr in low dword of destination)
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        addr = 0x80
        val64 = 0xABCD_EF01_2345_6789
        lds.write_qword(addr, val64)

        rf.write_vgpr(0, addr)               # v0 holds address
        rf.write_vgpr_64(2, 0xFFFF_FFFF_FFFF_FFFF)  # v[2:3] holds limit

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),  # v[0:1]
            vaddr=Register(RegKind.VGPR, 0, count=1), # v0
            vdata0=Register(RegKind.VGPR, 2, count=2),# v[2:3]
        )

        old, new_mem = interp.step(inst)
        self.assertEqual(old, val64)
        self.assertEqual(new_mem, val64 + 1)
        # v[0:1] now holds returned value val64
        self.assertEqual(rf.read_vgpr_64(0), val64)
        self.assertEqual(lds.read_qword(addr), val64 + 1)
        # v[2:3] remains unchanged
        self.assertEqual(rf.read_vgpr_64(2), 0xFFFF_FFFF_FFFF_FFFF)

    def test_alias_vdst_equals_vdata0(self) -> None:
        # vdst = v[0:1], vdata0 = v[0:1] (identical 64-bit register pair)
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        addr = 0xA0
        lds.write_qword(addr, 50)

        rf.write_vgpr(2, addr)        # v2 holds address
        rf.write_vgpr_64(0, 50)       # v[0:1] holds wrap limit (50 == 50 -> wrap)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 0, count=2),
        )

        old, new_mem = interp.step(inst)
        self.assertEqual(old, 50)
        self.assertEqual(new_mem, 0)
        # v[0:1] overwritten with old value 50
        self.assertEqual(rf.read_vgpr_64(0), 50)
        self.assertEqual(lds.read_qword(addr), 0)
        # v2 preserved
        self.assertEqual(rf.read_vgpr(2), addr)

    def test_alias_vaddr_shares_register_with_vdata0_low(self) -> None:
        # vaddr = v0, vdata0 = v[0:1], vdst = v[4:5]
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        # Let v0 = 0x80 (128). If v[0:1] = 0x80, wrap limit is 128
        addr_and_limit_lo = 0x80
        lds.write_qword(addr_and_limit_lo, 10)

        rf.write_vgpr(0, addr_and_limit_lo)
        rf.write_vgpr(1, 0)
        rf.write_vgpr_64(4, 0xCAFE_BABE)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 4, count=2),
            vaddr=Register(RegKind.VGPR, 0, count=1),
            vdata0=Register(RegKind.VGPR, 0, count=2),
        )

        old, new_mem = interp.step(inst)
        self.assertEqual(old, 10)
        self.assertEqual(new_mem, 11)
        self.assertEqual(rf.read_vgpr_64(4), 10)
        self.assertEqual(rf.read_vgpr(0), addr_and_limit_lo)
        self.assertEqual(lds.read_qword(addr_and_limit_lo), 11)

    def test_simd_aliasing_divergent_across_lanes(self) -> None:
        # Wave64 with alternating mask: vdst = v[0:1], vaddr = v0
        lds = LDSModel64(capacity_qwords=1024)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)
        simd.set_exec_mask(0x5555_5555_5555_5555)  # Even lanes active

        for lane in range(64):
            addr = 0x100 + lane * 8
            lds.write_qword(addr, 1000 + lane)
            simd.write_lane_vgpr(lane, 0, addr) # v0 is address
            simd.write_lane_vgpr_64(lane, 2, 5000)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),  # v[0:1]
            vaddr=Register(RegKind.VGPR, 0, count=1), # v0
            vdata0=Register(RegKind.VGPR, 2, count=2),# v[2:3]
        )

        results = simd.step(inst)
        for lane in range(64):
            addr = 0x100 + lane * 8
            if lane % 2 == 0:
                expected_old = 1000 + lane
                self.assertEqual(results[lane], (expected_old, expected_old + 1))
                self.assertEqual(simd.read_lane_vgpr_64(lane, 0), expected_old)
                self.assertEqual(lds.read_qword(addr), expected_old + 1)
            else:
                self.assertIsNone(results[lane])
                # Inactive lane preserves original address in v0
                self.assertEqual(simd.read_lane_vgpr(lane, 0), addr)
                self.assertEqual(lds.read_qword(addr), 1000 + lane)


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class AtomicInc64StatusFlags:
    """Architectural condition codes and status flags for ds_inc_rtn_u64."""
    wrapped: bool
    zero: bool
    negative: bool
    carry: bool
    borrow: bool
    threshold_equal: bool
    unaligned_addr: bool


def evaluate_ds_inc_64_flags(
    old_val: int,
    wrap_limit: int,
    eff_addr: int,
    new_mem: int,
) -> AtomicInc64StatusFlags:
    """Evaluate architectural flags for a 64-bit atomic increment with wrap operation."""
    u_old = u64(old_val)
    u_limit = u64(wrap_limit)

    wrapped = (u_old >= u_limit)
    zero = (u_old == 0)
    negative = bool(u_old & (1 << 63))
    borrow = (u_old < u_limit)
    threshold_equal = (u_old == u_limit)

    # Carry out generated by the 64-bit adder stage before the wrap multiplexer
    carry = bool((u_old + 1) > MASK_64)
    # Low 3 bits of byte address must be zero for aligned 64-bit qword access
    unaligned_addr = bool(eff_addr & 0x7)

    return AtomicInc64StatusFlags(
        wrapped=wrapped,
        zero=zero,
        negative=negative,
        carry=carry,
        borrow=borrow,
        threshold_equal=threshold_equal,
        unaligned_addr=unaligned_addr,
    )


class TestDsIncRtnU64ConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_normal_increment_flags_u64(self) -> None:
        flags = evaluate_ds_inc_64_flags(
            old_val=0x1_0000_0000,
            wrap_limit=0x2_0000_0000,
            eff_addr=0x100,
            new_mem=0x1_0000_0001,
        )
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.carry)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.threshold_equal)
        self.assertFalse(flags.unaligned_addr)

    def test_zero_flag_on_zero_old_val_u64(self) -> None:
        flags = evaluate_ds_inc_64_flags(
            old_val=0,
            wrap_limit=5,
            eff_addr=0x200,
            new_mem=1,
        )
        self.assertTrue(flags.zero)
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.borrow)

    def test_negative_flag_on_msb_set_u64(self) -> None:
        # Bit 63 set
        flags = evaluate_ds_inc_64_flags(
            old_val=0x8000_0000_0000_0000,
            wrap_limit=0xFFFF_FFFF_FFFF_FFFF,
            eff_addr=0x300,
            new_mem=0x8000_0000_0000_0001,
        )
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertFalse(flags.wrapped)
        self.assertTrue(flags.borrow)

    def test_wrap_threshold_equal_flags_u64(self) -> None:
        limit = 0x1234_5678_9ABC_DEF0
        flags = evaluate_ds_inc_64_flags(
            old_val=limit,
            wrap_limit=limit,
            eff_addr=0x400,
            new_mem=0,
        )
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.threshold_equal)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.carry)

    def test_wrap_overshoot_flags_u64(self) -> None:
        flags = evaluate_ds_inc_64_flags(
            old_val=100,
            wrap_limit=50,
            eff_addr=0x500,
            new_mem=0,
        )
        self.assertTrue(flags.wrapped)
        self.assertFalse(flags.threshold_equal)
        self.assertFalse(flags.borrow)

    def test_unaligned_qword_address_flag_detection(self) -> None:
        for offset in range(1, 8):
            flags = evaluate_ds_inc_64_flags(1, 10, 0x100 + offset, 2)
            self.assertTrue(flags.unaligned_addr)

        flags_aligned = evaluate_ds_inc_64_flags(1, 10, 0x108, 2)
        self.assertFalse(flags_aligned.unaligned_addr)

    def test_carry_generation_on_max_u64_adder(self) -> None:
        flags = evaluate_ds_inc_64_flags(
            old_val=0xFFFF_FFFF_FFFF_FFFF,
            wrap_limit=0xFFFF_FFFF_FFFF_FFFF,
            eff_addr=0x600,
            new_mem=0,
        )
        self.assertTrue(flags.carry)
        self.assertTrue(flags.wrapped)


# ==============================================================================
# Phase 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64 = 0x0000_0000_0000_0000
NEG_ZERO_F64 = 0x8000_0000_0000_0000
POS_INF_F64 = 0x7FF0_0000_0000_0000
NEG_INF_F64 = 0xFFF0_0000_0000_0000
QNAN_F64 = 0x7FF8_0000_0000_0000
SNAN_F64 = 0x7FF0_0000_0000_0001
MIN_NORMAL_F64 = 0x0010_0000_0000_0000
MIN_SUBNORMAL_F64 = 0x0000_0000_0000_0001
MAX_SUBNORMAL_F64 = 0x000F_FFFF_FFFF_FFFF
MAX_NORMAL_F64 = 0x7FEF_FFFF_FFFF_FFFF
ONE_F64 = 0x3FF0_0000_0000_0000


def bits_to_f64(bits: int) -> float:
    """Interpret 64-bit unsigned integer as IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK_64))[0]


def f64_to_bits(val: float) -> int:
    """Pack IEEE 754 double-precision float into 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_f64_nan(bits: int) -> bool:
    """Test if 64-bit bit pattern is an IEEE 754 double-precision NaN."""
    return (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def is_f64_subnormal(bits: int) -> bool:
    """Test if 64-bit bit pattern is an IEEE 754 double-precision subnormal float."""
    return (bits & 0x7FF0_0000_0000_0000) == 0 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


class TestDsIncRtnU64IEEE754BoundaryExtremes(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_f64_increment_no_flush_to_zero(self) -> None:
        # Subnormals are not flushed to zero by LDS bitwise integer atomics
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        addr = 0x100
        lds.write_qword(addr, MIN_SUBNORMAL_F64)
        rf.write_vgpr(2, addr)
        rf.write_vgpr_64(4, 0xFFFF_FFFF_FFFF_FFFF)

        self.assertTrue(is_f64_subnormal(MIN_SUBNORMAL_F64))

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        old, mem = interp.step(inst)
        self.assertEqual(old, MIN_SUBNORMAL_F64)
        self.assertEqual(mem, MIN_SUBNORMAL_F64 + 1)
        self.assertNotEqual(mem, 0)
        self.assertTrue(is_f64_subnormal(mem))
        self.assertEqual(rf.read_vgpr_64(0), MIN_SUBNORMAL_F64)

    def test_subnormal_to_normal_boundary_crossing_f64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        addr = 0x120
        lds.write_qword(addr, MAX_SUBNORMAL_F64)
        rf.write_vgpr(2, addr)
        rf.write_vgpr_64(4, 0xFFFF_FFFF_FFFF_FFFF)

        self.assertTrue(is_f64_subnormal(MAX_SUBNORMAL_F64))

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        old, mem = interp.step(inst)
        self.assertEqual(old, MAX_SUBNORMAL_F64)
        self.assertEqual(mem, MIN_NORMAL_F64)
        self.assertFalse(is_f64_subnormal(mem))
        self.assertEqual(rf.read_vgpr_64(0), MAX_SUBNORMAL_F64)

    def test_signaling_nan_preserved_without_quietening_f64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        addr = 0x140
        lds.write_qword(addr, SNAN_F64)
        rf.write_vgpr(2, addr)
        rf.write_vgpr_64(4, 0xFFFF_FFFF_FFFF_FFFF)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        old, mem = interp.step(inst)
        self.assertEqual(old, SNAN_F64)
        self.assertEqual(mem, SNAN_F64 + 1)
        # Verify quiet bit 51 is NOT forced high
        self.assertEqual(mem & 0x0008_0000_0000_0000, 0)
        self.assertTrue(is_f64_nan(mem))
        self.assertEqual(rf.read_vgpr_64(0), SNAN_F64)

    def test_quiet_nan_wrap_boundary_f64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        addr = 0x160
        lds.write_qword(addr, QNAN_F64)
        rf.write_vgpr(2, addr)
        rf.write_vgpr_64(4, QNAN_F64)

        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )

        old, mem = interp.step(inst)
        self.assertEqual(old, QNAN_F64)
        self.assertEqual(mem, 0)
        self.assertEqual(rf.read_vgpr_64(0), QNAN_F64)
        self.assertEqual(lds.read_qword(addr), 0)

    def test_positive_and_negative_zero_asymmetry_f64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        # POS_ZERO_F64 is integer 0
        lds.write_qword(0x200, POS_ZERO_F64)
        rf.write_vgpr(2, 0x200)
        rf.write_vgpr_64(4, 100)
        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )
        old, mem = interp.step(inst)
        self.assertEqual(old, 0)
        self.assertEqual(mem, 1)

        # NEG_ZERO_F64 is integer 0x8000_0000_0000_0000
        lds.write_qword(0x208, NEG_ZERO_F64)
        rf.write_vgpr(2, 0x208)
        rf.write_vgpr_64(4, 0xFFFF_FFFF_FFFF_FFFF)
        old2, mem2 = interp.step(inst)
        self.assertEqual(old2, 0x8000_0000_0000_0000)
        self.assertEqual(mem2, 0x8000_0000_0000_0001)

    def test_infinities_bit_patterns_f64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        # POS_INF_F64 is 0x7FF0_0000_0000_0000
        addr = 0x300
        lds.write_qword(addr, POS_INF_F64)
        rf.write_vgpr(2, addr)
        rf.write_vgpr_64(4, 0xFFFF_FFFF_FFFF_FFFF)
        inst = Ds64Instruction(
            opcode=Ds64Opcode.DS_INC_RTN_U64,
            vdst=Register(RegKind.VGPR, 0, count=2),
            vaddr=Register(RegKind.VGPR, 2, count=1),
            vdata0=Register(RegKind.VGPR, 4, count=2),
        )
        old, mem = interp.step(inst)
        self.assertEqual(old, POS_INF_F64)
        self.assertEqual(mem, POS_INF_F64 + 1)

        # NEG_INF_F64 is 0xFFF0_0000_0000_0000
        addr2 = 0x308
        lds.write_qword(addr2, NEG_INF_F64)
        rf.write_vgpr(2, addr2)
        rf.write_vgpr_64(4, NEG_INF_F64)
        old2, mem2 = interp.step(inst)
        self.assertEqual(old2, NEG_INF_F64)
        self.assertEqual(mem2, 0)

    def test_uint64_boundary_extremes(self) -> None:
        u_max = 0xFFFF_FFFF_FFFF_FFFF
        u_mid = 0x8000_0000_0000_0000
        u_mid_minus = 0x7FFF_FFFF_FFFF_FFFF

        old, mem = oracle_ds_inc_rtn_u64(u_max - 1, u_max)
        self.assertEqual(old, u_max - 1)
        self.assertEqual(mem, u_max)

        old, mem = oracle_ds_inc_rtn_u64(u_max, u_max)
        self.assertEqual(old, u_max)
        self.assertEqual(mem, 0)

        old, mem = oracle_ds_inc_rtn_u64(u_mid_minus, u_max)
        self.assertEqual(old, u_mid_minus)
        self.assertEqual(mem, u_mid)

    def test_counter_cyclic_trajectory_u64(self) -> None:
        limit = 5
        curr = 0
        for expected in range(limit + 1):
            old, curr = oracle_ds_inc_rtn_u64(curr, limit)
            self.assertEqual(old, expected)
            if expected < limit:
                self.assertEqual(curr, expected + 1)
            else:
                self.assertEqual(curr, 0)
        self.assertEqual(curr, 0)


if __name__ == "__main__":
    unittest.main()
