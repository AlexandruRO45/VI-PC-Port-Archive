from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_inc_rtn_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Increment with Return:
Performs a 32-bit atomic wrapping increment on LDS memory and returns the original value.

Syntax:
    ds_inc_rtn_u32 vdst, vaddr, vdata0 [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    new_mem = 0 if old_val >= (vdata0 & 0xFFFFFFFF) else (old_val + 1) & 0xFFFFFFFF
    LDS[dword_index] = new_mem
    vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing:
      Base byte address provided in VGPR vaddr.
      Optional 16-bit unsigned byte offset added to base address:
        eff_addr = (vaddr + offset16) mod 2^32.
      Addresses are dword-aligned: low 2 bits are dropped (eff_addr >> 2).
      Dword indices wrap modulo the allocated power-of-two capacity.
  - Wrap Condition:
      Unsigned 32-bit integer comparison between current memory value (old_val)
      and the wrap limit operand (vdata0).
      When old_val >= wrap_limit, memory wraps to 0.
      Otherwise, memory is incremented by 1 mod 2^32.
  - Return:
      The pre-operation memory value (old_val) is returned to vector register vdst.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
import struct
import unittest
from typing import Optional, Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask
DEFAULT_NUM_VGPRS: int = 256        # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106        # Standard GFX10 scalar register file size
DEFAULT_NUM_LANES: int = 64         # Wave64 default width


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK_32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_inc_rtn_u32(
    current_mem: int,
    wrap_limit: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_inc_rtn_u32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        limit = wrap_limit & 0xFFFFFFFF
        new_mem = 0 if old_val >= limit else (old_val + 1) & 0xFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    limit = u32(wrap_limit)
    new_mem = 0 if old_val >= limit else u32(old_val + 1)
    return old_val, new_mem


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a positive power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = u32(val)

    def atomic_inc_rtn_u32(
        self,
        vaddr: int,
        wrap_limit: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit wrapping increment with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_inc_rtn_u32(curr, wrap_limit)
        self.memory[idx] = new_mem
        return old_val, new_mem


class TestReferenceOracleAndLDSModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_normal_increment(self) -> None:
        # Incrementing below wrap limit returns old value and increments memory
        old, new_mem = oracle_ds_inc_rtn_u32(0, 5)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 1)

        old, new_mem = oracle_ds_inc_rtn_u32(1, 5)
        self.assertEqual(old, 1)
        self.assertEqual(new_mem, 2)

        old, new_mem = oracle_ds_inc_rtn_u32(4, 5)
        self.assertEqual(old, 4)
        self.assertEqual(new_mem, 5)

    def test_oracle_wrap_on_equal_threshold(self) -> None:
        # Equal to threshold wraps memory to 0, returns threshold
        old, new_mem = oracle_ds_inc_rtn_u32(5, 5)
        self.assertEqual(old, 5)
        self.assertEqual(new_mem, 0)

    def test_oracle_wrap_on_greater_threshold(self) -> None:
        # Strictly exceeding threshold wraps memory to 0, returns old value
        old, new_mem = oracle_ds_inc_rtn_u32(6, 5)
        self.assertEqual(old, 6)
        self.assertEqual(new_mem, 0)

        old, new_mem = oracle_ds_inc_rtn_u32(100, 5)
        self.assertEqual(old, 100)
        self.assertEqual(new_mem, 0)

    def test_oracle_zero_wrap_limit(self) -> None:
        # Limit 0: any unsigned value >= 0 wraps to 0
        old, new_mem = oracle_ds_inc_rtn_u32(0, 0)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 0)

        old, new_mem = oracle_ds_inc_rtn_u32(10, 0)
        self.assertEqual(old, 10)
        self.assertEqual(new_mem, 0)

    def test_oracle_max_u32_limit(self) -> None:
        # Limit 0xFFFFFFFF
        old, new_mem = oracle_ds_inc_rtn_u32(0, MASK_32)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 1)

        old, new_mem = oracle_ds_inc_rtn_u32(MASK_32 - 1, MASK_32)
        self.assertEqual(old, MASK_32 - 1)
        self.assertEqual(new_mem, MASK_32)

        old, new_mem = oracle_ds_inc_rtn_u32(MASK_32, MASK_32)
        self.assertEqual(old, MASK_32)
        self.assertEqual(new_mem, 0)

    def test_effective_address_calculation_and_wrapping(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x0024)
        self.assertEqual(eff, 0x1024)

        # Modulo 2^32 address wrapping
        eff_wrap = calc_lds_effective_address(0xFFFF_FFF0, 0x0020)
        self.assertEqual(eff_wrap, 0x0000_0010)

        # 16-bit offset masking
        eff_offset_mask = calc_lds_effective_address(0x2000, 0x1_0004)
        self.assertEqual(eff_offset_mask, 0x2004)

    def test_dword_alignment_and_low_bits_truncation(self) -> None:
        # Low 2 bits are dropped (dword alignment)
        self.assertEqual(calc_lds_dword_index(0x1000), 0x400)
        self.assertEqual(calc_lds_dword_index(0x1001), 0x400)
        self.assertEqual(calc_lds_dword_index(0x1002), 0x400)
        self.assertEqual(calc_lds_dword_index(0x1003), 0x400)
        self.assertEqual(calc_lds_dword_index(0x1004), 0x401)

    def test_capacity_wrapping(self) -> None:
        capacity_bytes = LDS_SIZE_DWORDS * 4
        eff = calc_lds_effective_address(capacity_bytes + 0x40)
        idx = calc_lds_dword_index(eff, LDS_SIZE_DWORDS)
        self.assertEqual(idx, 0x10)

    def test_bank_distribution(self) -> None:
        for bank in range(LDS_BANK_COUNT):
            addr = bank * 4
            self.assertEqual(calc_lds_bank(addr), bank)

    def test_lds_model_atomic_inc_rtn(self) -> None:
        lds = LDSModel(capacity_dwords=1024)
        addr = 0x100  # dword index 64

        lds.write_dword(addr, 2)
        old, new_mem = lds.atomic_inc_rtn_u32(vaddr=addr, wrap_limit=5, offset=0)
        self.assertEqual(old, 2)
        self.assertEqual(new_mem, 3)
        self.assertEqual(lds.read_dword(addr), 3)

        # Increment with offset
        old, new_mem = lds.atomic_inc_rtn_u32(vaddr=0xFC, wrap_limit=5, offset=4)
        self.assertEqual(old, 3)
        self.assertEqual(new_mem, 4)
        self.assertEqual(lds.read_dword(addr), 4)

        # Hit wrap condition
        old, new_mem = lds.atomic_inc_rtn_u32(vaddr=addr, wrap_limit=4, offset=0)
        self.assertEqual(old, 4)
        self.assertEqual(new_mem, 0)
        self.assertEqual(lds.read_dword(addr), 0)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_INC_RTN_U32 = "ds_inc_rtn_u32"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"{self.reg_type.name} index out of architectural bounds [0..{max_limit - 1}]: {self.index}"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class DSIncRtnInstruction:
    """Symbolic AST node for ds_inc_rtn_u32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata: RegisterOperand
    offset: int = 0
    opcode: DSOpcode = DSOpcode.DS_INC_RTN_U32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise TypeError(f"vdst must be VGPR, got {self.vdst.reg_type}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise TypeError(f"vaddr must be VGPR, got {self.vaddr.reg_type}")
        if self.vdata.reg_type != RegType.VGPR:
            raise TypeError(f"vdata must be VGPR, got {self.vdata.reg_type}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of bounds [0..0xFFFF]: {self.offset}")

    def format_asm(self) -> str:
        base = f"{self.opcode.value} {self.vdst.name}, {self.vaddr.name}, {self.vdata.name}"
        if self.offset != 0:
            base += f" offset:{self.offset}"
        return base

    @classmethod
    def parse(cls, asm_line: str) -> DSIncRtnInstruction:
        clean = re.sub(r"//.*|/\*.*?\*/|;.*", "", asm_line).strip()
        if not clean:
            raise ValueError("Empty assembly line")

        tokens = [t.strip() for t in clean.replace(",", " ").split() if t.strip()]
        if len(tokens) < 4:
            raise ValueError(f"Too few tokens for ds_inc_rtn_u32 instruction: '{clean}'")

        op_str = tokens[0].lower()
        if op_str != DSOpcode.DS_INC_RTN_U32.value:
            raise ValueError(f"Expected opcode {DSOpcode.DS_INC_RTN_U32.value}, got: '{tokens[0]}'")

        def parse_vgpr(reg_str: str) -> RegisterOperand:
            m = re.match(r"^v(\d+)$", reg_str.lower())
            if not m:
                raise ValueError(f"Expected VGPR token (e.g. v0), got: '{reg_str}'")
            return RegisterOperand(int(m.group(1)), RegType.VGPR)

        vdst = parse_vgpr(tokens[1])
        vaddr = parse_vgpr(tokens[2])
        vdata = parse_vgpr(tokens[3])
        offset = 0

        for extra in tokens[4:]:
            m = re.match(r"^offset:(\d+|0x[0-9a-fA-F]+)$", extra.lower())
            if m:
                val_str = m.group(1)
                offset = int(val_str, 16 if val_str.startswith("0x") else 10)
            else:
                raise ValueError(f"Unrecognized instruction modifier: '{extra}'")

        return cls(vdst=vdst, vaddr=vaddr, vdata=vdata, offset=offset)


class TestSymbolicASTAndInstruction(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v1 = RegisterOperand(1, RegType.VGPR)
        v2 = RegisterOperand(2, RegType.VGPR)
        instr = DSIncRtnInstruction(vdst=v0, vaddr=v1, vdata=v2, offset=16)

        self.assertEqual(instr.opcode, DSOpcode.DS_INC_RTN_U32)
        self.assertEqual(instr.vdst.name, "v0")
        self.assertEqual(instr.vaddr.name, "v1")
        self.assertEqual(instr.vdata.name, "v2")
        self.assertEqual(instr.offset, 16)

    def test_asm_formatting(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v1 = RegisterOperand(1, RegType.VGPR)
        v2 = RegisterOperand(2, RegType.VGPR)

        instr_no_offset = DSIncRtnInstruction(vdst=v0, vaddr=v1, vdata=v2, offset=0)
        self.assertEqual(instr_no_offset.format_asm(), "ds_inc_rtn_u32 v0, v1, v2")

        instr_offset = DSIncRtnInstruction(vdst=v0, vaddr=v1, vdata=v2, offset=32)
        self.assertEqual(instr_offset.format_asm(), "ds_inc_rtn_u32 v0, v1, v2 offset:32")

    def test_parse_valid_asm_lines(self) -> None:
        instr1 = DSIncRtnInstruction.parse("ds_inc_rtn_u32 v0, v1, v2")
        self.assertEqual(instr1.vdst.index, 0)
        self.assertEqual(instr1.vaddr.index, 1)
        self.assertEqual(instr1.vdata.index, 2)
        self.assertEqual(instr1.offset, 0)

        instr2 = DSIncRtnInstruction.parse("ds_inc_rtn_u32 v10, v20, v30 offset:64 ; with comment")
        self.assertEqual(instr2.vdst.index, 10)
        self.assertEqual(instr2.vaddr.index, 20)
        self.assertEqual(instr2.vdata.index, 30)
        self.assertEqual(instr2.offset, 64)

        instr3 = DSIncRtnInstruction.parse("ds_inc_rtn_u32 v5, v6, v7 offset:0x20 // hex offset")
        self.assertEqual(instr3.offset, 32)

    def test_register_bounds_and_type_validation(self) -> None:
        # Valid bounds
        RegisterOperand(0, RegType.VGPR)
        RegisterOperand(255, RegType.VGPR)
        RegisterOperand(105, RegType.SGPR)

        # Out of bounds
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

        # SGPR invalid for LDS vector instructions
        s0 = RegisterOperand(0, RegType.SGPR)
        v1 = RegisterOperand(1, RegType.VGPR)
        v2 = RegisterOperand(2, RegType.VGPR)
        with self.assertRaises(TypeError):
            DSIncRtnInstruction(vdst=s0, vaddr=v1, vdata=v2)
        with self.assertRaises(TypeError):
            DSIncRtnInstruction(vdst=v1, vaddr=s0, vdata=v2)
        with self.assertRaises(TypeError):
            DSIncRtnInstruction(vdst=v1, vaddr=v2, vdata=s0)

        # Offset validation
        with self.assertRaises(ValueError):
            DSIncRtnInstruction(vdst=v1, vaddr=v1, vdata=v2, offset=-1)
        with self.assertRaises(ValueError):
            DSIncRtnInstruction(vdst=v1, vaddr=v1, vdata=v2, offset=0x10000)

    def test_parse_rejections(self) -> None:
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("")
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_inc_rtn_u32 v0, v1")  # too few tokens
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_add_rtn_u32 v0, v1, v2")  # wrong opcode
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_inc_rtn_u32 s0, v1, v2")  # not VGPR
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_inc_rtn_u32 v0, v1, v2 bad_modifier")


# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState:
    """Architectural state representation for single-thread scalar execution."""

    def __init__(
        self,
        lds_capacity_dwords: int = LDS_SIZE_DWORDS,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
        num_sgprs: int = DEFAULT_NUM_SGPRS,
    ) -> None:
        self.lds = LDSModel(capacity_dwords=lds_capacity_dwords)
        self.num_vgprs = num_vgprs
        self.num_sgprs = num_sgprs
        self.vgprs: list[int] = [0] * num_vgprs
        self.sgprs: list[int] = [0] * num_sgprs
        self.m0: int = 0

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index < self.num_vgprs):
            raise IndexError(f"VGPR index out of range [0..{self.num_vgprs - 1}]: {index}")
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index < self.num_vgprs):
            raise IndexError(f"VGPR index out of range [0..{self.num_vgprs - 1}]: {index}")
        self.vgprs[index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index < self.num_sgprs):
            raise IndexError(f"SGPR index out of range [0..{self.num_sgprs - 1}]: {index}")
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index < self.num_sgprs):
            raise IndexError(f"SGPR index out of range [0..{self.num_sgprs - 1}]: {index}")
        self.sgprs[index] = u32(val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter executing ds_inc_rtn_u32."""

    def step(
        self,
        instr: DSIncRtnInstruction,
        state: ScalarMachineState,
    ) -> Tuple[int, int]:
        """Execute one ds_inc_rtn_u32 instruction sequentially.

        Returns:
            Tuple[int, int]: (old_val, new_mem)
        """
        vaddr = state.get_vgpr(instr.vaddr.index)
        wrap_limit = state.get_vgpr(instr.vdata.index)

        old_val, new_mem = state.lds.atomic_inc_rtn_u32(
            vaddr=vaddr,
            wrap_limit=wrap_limit,
            offset=instr.offset,
        )

        state.set_vgpr(instr.vdst.index, old_val)
        return old_val, new_mem

    def parse_and_step(
        self,
        asm_line: str,
        state: ScalarMachineState,
    ) -> Tuple[int, int]:
        """Parse assembly text and execute one step."""
        instr = DSIncRtnInstruction.parse(asm_line)
        return self.step(instr, state)


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def test_single_step_execution(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x200
        state.lds.write_dword(addr, 10)
        state.set_vgpr(1, addr)   # vaddr = 0x200
        state.set_vgpr(2, 20)     # vdata = wrap_limit 20
        state.set_vgpr(0, 0xDEAD) # vdst

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )

        old_val, new_mem = interp.step(instr, state)

        self.assertEqual(old_val, 10)
        self.assertEqual(new_mem, 11)
        self.assertEqual(state.get_vgpr(0), 10)
        self.assertEqual(state.lds.read_dword(addr), 11)

    def test_wrapping_cycle(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x100
        state.lds.write_dword(addr, 0)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 3)  # limit = 3: 0 -> 1 -> 2 -> 3 -> 0

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        # Step 1: 0 -> 1 (old 0)
        old, mem = interp.step(instr, state)
        self.assertEqual((old, mem), (0, 1))
        self.assertEqual(state.get_vgpr(0), 0)

        # Step 2: 1 -> 2 (old 1)
        old, mem = interp.step(instr, state)
        self.assertEqual((old, mem), (1, 2))
        self.assertEqual(state.get_vgpr(0), 1)

        # Step 3: 2 -> 3 (old 2)
        old, mem = interp.step(instr, state)
        self.assertEqual((old, mem), (2, 3))
        self.assertEqual(state.get_vgpr(0), 2)

        # Step 4: 3 -> 0 (wrap! old 3)
        old, mem = interp.step(instr, state)
        self.assertEqual((old, mem), (3, 0))
        self.assertEqual(state.get_vgpr(0), 3)

        # Step 5: 0 -> 1 (old 0)
        old, mem = interp.step(instr, state)
        self.assertEqual((old, mem), (0, 1))
        self.assertEqual(state.get_vgpr(0), 0)

    def test_offset_immediate_stepping(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        state.lds.write_dword(0x310, 50)
        state.set_vgpr(1, 0x300)  # base
        state.set_vgpr(2, 100)    # limit

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
            offset=0x10,
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, 50)
        self.assertEqual(mem, 51)
        self.assertEqual(state.lds.read_dword(0x310), 51)

    def test_parse_and_step_interface(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        state.lds.write_dword(0x40, 7)
        state.set_vgpr(1, 0x40)
        state.set_vgpr(2, 10)

        old, mem = interp.parse_and_step("ds_inc_rtn_u32 v0, v1, v2", state)
        self.assertEqual(old, 7)
        self.assertEqual(mem, 8)
        self.assertEqual(state.get_vgpr(0), 7)

    def test_untouched_registers_preserved(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        state.set_vgpr(1, 0x80)
        state.set_vgpr(2, 10)
        state.set_vgpr(3, 0xCAFE)
        state.set_vgpr(4, 0xBEEF)

        interp.parse_and_step("ds_inc_rtn_u32 v0, v1, v2", state)
        self.assertEqual(state.get_vgpr(3), 0xCAFE)
        self.assertEqual(state.get_vgpr(4), 0xBEEF)


# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDState:
    """Architectural SIMD wavefront state for Wave32 and Wave64 execution."""

    def __init__(
        self,
        num_lanes: int = DEFAULT_NUM_LANES,
        exec_mask: int = MASK_32,
        lds_capacity_dwords: int = LDS_SIZE_DWORDS,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got: {num_lanes}")
        self.num_lanes = num_lanes
        self.mask_limit = 0xFFFFFFFF if num_lanes == 32 else 0xFFFFFFFFFFFFFFFF
        self.exec_mask = exec_mask & self.mask_limit
        self.num_vgprs = num_vgprs
        self.lds = LDSModel(capacity_dwords=lds_capacity_dwords)
        self.vgprs: list[list[int]] = [[0] * num_vgprs for _ in range(num_lanes)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane index out of bounds [0..{self.num_lanes - 1}]: {lane}")
        return bool((self.exec_mask >> lane) & 1)

    def set_lane_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane index out of bounds [0..{self.num_lanes - 1}]: {lane}")
        if not (0 <= reg_idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of bounds [0..{self.num_vgprs - 1}]: {reg_idx}")
        self.vgprs[lane][reg_idx] = u32(val)

    def get_lane_vgpr(self, lane: int, reg_idx: int) -> int:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane index out of bounds [0..{self.num_lanes - 1}]: {lane}")
        if not (0 <= reg_idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of bounds [0..{self.num_vgprs - 1}]: {reg_idx}")
        return self.vgprs[lane][reg_idx]


class SIMDInterpreter:
    """Multi-lane SIMD interpreter executing ds_inc_rtn_u32 across wavefronts."""

    def execute(
        self,
        instr: DSIncRtnInstruction,
        state: SIMDState,
    ) -> list[Optional[Tuple[int, int]]]:
        """Execute ds_inc_rtn_u32 across all active lanes with LDS arbitration."""
        results: list[Optional[Tuple[int, int]]] = [None] * state.num_lanes

        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue

            vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
            wrap_limit = state.get_lane_vgpr(lane, instr.vdata.index)

            old_val, new_mem = state.lds.atomic_inc_rtn_u32(
                vaddr=vaddr,
                wrap_limit=wrap_limit,
                offset=instr.offset,
            )

            state.set_lane_vgpr(lane, instr.vdst.index, old_val)
            results[lane] = (old_val, new_mem)

        return results


class TestSIMDMultiLaneAndExecMasking(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds_capacity_dwords=1024)
        interp = SIMDInterpreter()

        # Each lane operates on its own dword
        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(addr, lane * 10)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 1000)
            state.set_lane_vgpr(lane, 0, 0xCAFE)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            self.assertEqual(results[lane], (lane * 10, lane * 10 + 1))
            self.assertEqual(state.get_lane_vgpr(lane, 0), lane * 10)
            self.assertEqual(state.lds.read_dword(lane * 4), lane * 10 + 1)

    def test_wave64_divergent_even_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask, lds_capacity_dwords=1024)
        interp = SIMDInterpreter()

        sentinel = 0xBAADF00D
        for lane in range(64):
            addr = lane * 4
            state.lds.write_dword(addr, 100)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 200)
            state.set_lane_vgpr(lane, 0, sentinel)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            if lane % 2 == 0:
                # Active
                self.assertEqual(results[lane], (100, 101))
                self.assertEqual(state.get_lane_vgpr(lane, 0), 100)
                self.assertEqual(state.lds.read_dword(lane * 4), 101)
            else:
                # Inactive: untouched
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 0), sentinel)
                self.assertEqual(state.lds.read_dword(lane * 4), 100)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0, lds_capacity_dwords=1024)
        interp = SIMDInterpreter()

        addr = 0x40
        state.lds.write_dword(addr, 55)
        for lane in range(64):
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 100)
            state.set_lane_vgpr(lane, 0, 77)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_lane_vgpr(lane, 0), 77)
        self.assertEqual(state.lds.read_dword(addr), 55)

    def test_single_lane_63_active(self) -> None:
        single_mask = 1 << 63
        state = SIMDState(num_lanes=64, exec_mask=single_mask, lds_capacity_dwords=1024)
        interp = SIMDInterpreter()

        addr = 0x180
        state.lds.write_dword(addr, 40)
        state.set_lane_vgpr(63, 1, addr)
        state.set_lane_vgpr(63, 2, 50)
        state.set_lane_vgpr(63, 0, 0)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(63):
            self.assertIsNone(results[lane])
        self.assertEqual(results[63], (40, 41))
        self.assertEqual(state.get_lane_vgpr(63, 0), 40)
        self.assertEqual(state.lds.read_dword(addr), 41)

    def test_multi_lane_lds_address_collision_serialization(self) -> None:
        # Lanes 0..3 active, all target the exact same address 0x80
        state = SIMDState(num_lanes=32, exec_mask=0xF, lds_capacity_dwords=1024)
        interp = SIMDInterpreter()

        common_addr = 0x80
        state.lds.write_dword(common_addr, 0)
        for lane in range(4):
            state.set_lane_vgpr(lane, 1, common_addr)
            state.set_lane_vgpr(lane, 2, 100) # limit 100
            state.set_lane_vgpr(lane, 0, 0)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        # Lane 0 sees 0, writes 1
        self.assertEqual(results[0], (0, 1))
        self.assertEqual(state.get_lane_vgpr(0, 0), 0)

        # Lane 1 sees 1, writes 2
        self.assertEqual(results[1], (1, 2))
        self.assertEqual(state.get_lane_vgpr(1, 0), 1)

        # Lane 2 sees 2, writes 3
        self.assertEqual(results[2], (2, 3))
        self.assertEqual(state.get_lane_vgpr(2, 0), 2)

        # Lane 3 sees 3, writes 4
        self.assertEqual(results[3], (3, 4))
        self.assertEqual(state.get_lane_vgpr(3, 0), 3)

        self.assertEqual(state.lds.read_dword(common_addr), 4)

    def test_multi_lane_collision_with_wrapping(self) -> None:
        # Lanes 0..3 all targeting common address with wrap limit = 2
        state = SIMDState(num_lanes=32, exec_mask=0xF, lds_capacity_dwords=1024)
        interp = SIMDInterpreter()

        common_addr = 0xC0
        state.lds.write_dword(common_addr, 0)
        for lane in range(4):
            state.set_lane_vgpr(lane, 1, common_addr)
            state.set_lane_vgpr(lane, 2, 2)  # limit 2: 0 -> 1 -> 2 -> 0 -> 1
            state.set_lane_vgpr(lane, 0, 0)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        # Lane 0: old 0, mem becomes 1
        self.assertEqual(results[0], (0, 1))
        # Lane 1: old 1, mem becomes 2
        self.assertEqual(results[1], (1, 2))
        # Lane 2: old 2, wraps to 0
        self.assertEqual(results[2], (2, 0))
        # Lane 3: old 0, mem becomes 1
        self.assertEqual(results[3], (0, 1))

        self.assertEqual(state.lds.read_dword(common_addr), 1)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_vdst_equals_vaddr(self) -> None:
        # ds_inc_rtn_u32 v0, v0, v1 (vdst == vaddr)
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x100
        state.lds.write_dword(addr, 42)
        state.set_vgpr(0, addr)  # v0 is vaddr
        state.set_vgpr(1, 100)   # v1 is wrap_limit

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata=RegisterOperand(1, RegType.VGPR),
            offset=0,
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, 42)
        self.assertEqual(mem, 43)
        # v0 now holds the returned old value 42
        self.assertEqual(state.get_vgpr(0), 42)
        self.assertEqual(state.lds.read_dword(addr), 43)
        # v1 (wrap_limit) must be intact
        self.assertEqual(state.get_vgpr(1), 100)

    def test_alias_vdst_equals_vdata(self) -> None:
        # ds_inc_rtn_u32 v1, v0, v1 (vdst == vdata)
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x140
        state.lds.write_dword(addr, 5)
        state.set_vgpr(0, addr) # v0 is vaddr
        state.set_vgpr(1, 5)    # v1 is wrap_limit (5 == 5 -> wrap)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(1, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata=RegisterOperand(1, RegType.VGPR),
            offset=0,
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, 5)
        self.assertEqual(mem, 0)
        # Memory wrapped to 0
        self.assertEqual(state.lds.read_dword(addr), 0)
        # v1 overwritten with old value 5
        self.assertEqual(state.get_vgpr(1), 5)
        # v0 preserved
        self.assertEqual(state.get_vgpr(0), addr)

    def test_alias_vaddr_equals_vdata(self) -> None:
        # ds_inc_rtn_u32 v2, v0, v0 (vaddr == vdata)
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        # v0 holds 0x80 (128): address is 0x80, wrap_limit is 128
        addr_and_limit = 0x80
        state.lds.write_dword(addr_and_limit, 10)
        state.set_vgpr(0, addr_and_limit)
        state.set_vgpr(2, 0xCAFE)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(2, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata=RegisterOperand(0, RegType.VGPR),
            offset=0,
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, 10)
        self.assertEqual(mem, 11)
        self.assertEqual(state.get_vgpr(2), 10)
        # v0 is untouched
        self.assertEqual(state.get_vgpr(0), addr_and_limit)
        self.assertEqual(state.lds.read_dword(addr_and_limit), 11)

    def test_alias_three_way_all_identical(self) -> None:
        # ds_inc_rtn_u32 v0, v0, v0 (vdst == vaddr == vdata)
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr_val = 0x40  # 64
        state.lds.write_dword(addr_val, 3)
        state.set_vgpr(0, addr_val)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata=RegisterOperand(0, RegType.VGPR),
            offset=0,
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, 3)
        self.assertEqual(mem, 4)
        # Memory updated to 4
        self.assertEqual(state.lds.read_dword(addr_val), 4)
        # v0 overwritten with old value 3
        self.assertEqual(state.get_vgpr(0), 3)

    def test_simd_divergent_aliased_operand_preservation(self) -> None:
        # Wave64 with alternating mask: vdst == vaddr
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask, lds_capacity_dwords=1024)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x200 + lane * 4
            state.lds.write_dword(addr, 20 + lane)
            state.set_lane_vgpr(lane, 0, addr) # v0 is vaddr and vdst
            state.set_lane_vgpr(lane, 1, 100)  # v1 is limit

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata=RegisterOperand(1, RegType.VGPR),
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            addr = 0x200 + lane * 4
            if lane % 2 == 0:
                # Active: v0 updated to old value (20 + lane)
                self.assertEqual(results[lane], (20 + lane, 20 + lane + 1))
                self.assertEqual(state.get_lane_vgpr(lane, 0), 20 + lane)
                self.assertEqual(state.lds.read_dword(addr), 20 + lane + 1)
            else:
                # Inactive: v0 preserves original address (addr)
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 0), addr)
                self.assertEqual(state.lds.read_dword(addr), 20 + lane)


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class AtomicIncStatusFlags:
    """Architectural condition codes and status flags for ds_inc_rtn_u32."""
    wrapped: bool
    zero: bool
    negative: bool
    carry: bool
    borrow: bool
    threshold_equal: bool
    unaligned_addr: bool


def evaluate_ds_inc_flags(
    old_val: int,
    wrap_limit: int,
    eff_addr: int,
    new_mem: int,
) -> AtomicIncStatusFlags:
    """Evaluate architectural flags for an atomic increment with wrap operation."""
    u_old = u32(old_val)
    u_limit = u32(wrap_limit)

    wrapped = (u_old >= u_limit)
    zero = (u_old == 0)
    negative = bool(u_old & 0x8000_0000)
    borrow = (u_old < u_limit)
    threshold_equal = (u_old == u_limit)

    # Carry out generated by the 32-bit adder stage before the wrap multiplexer
    carry = bool((u_old + 1) > MASK_32)
    unaligned_addr = bool(eff_addr & 0x3)

    return AtomicIncStatusFlags(
        wrapped=wrapped,
        zero=zero,
        negative=negative,
        carry=carry,
        borrow=borrow,
        threshold_equal=threshold_equal,
        unaligned_addr=unaligned_addr,
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_normal_increment_flags(self) -> None:
        flags = evaluate_ds_inc_flags(
            old_val=10,
            wrap_limit=20,
            eff_addr=0x100,
            new_mem=11,
        )
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.carry)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.threshold_equal)
        self.assertFalse(flags.unaligned_addr)

    def test_zero_flag_on_zero_old_val(self) -> None:
        flags = evaluate_ds_inc_flags(
            old_val=0,
            wrap_limit=5,
            eff_addr=0x200,
            new_mem=1,
        )
        self.assertTrue(flags.zero)
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.borrow)

    def test_negative_flag_on_msb_set(self) -> None:
        flags = evaluate_ds_inc_flags(
            old_val=0x8000_0000,
            wrap_limit=0xFFFF_FFFF,
            eff_addr=0x300,
            new_mem=0x8000_0001,
        )
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertFalse(flags.wrapped)
        self.assertTrue(flags.borrow)

    def test_wrap_threshold_equal_flags(self) -> None:
        flags = evaluate_ds_inc_flags(
            old_val=50,
            wrap_limit=50,
            eff_addr=0x400,
            new_mem=0,
        )
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.threshold_equal)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.carry)

    def test_wrap_overshoot_flags(self) -> None:
        flags = evaluate_ds_inc_flags(
            old_val=100,
            wrap_limit=50,
            eff_addr=0x500,
            new_mem=0,
        )
        self.assertTrue(flags.wrapped)
        self.assertFalse(flags.threshold_equal)
        self.assertFalse(flags.borrow)

    def test_unaligned_address_flag_detection(self) -> None:
        flags_unaligned_1 = evaluate_ds_inc_flags(1, 10, 0x101, 2)
        flags_unaligned_2 = evaluate_ds_inc_flags(1, 10, 0x102, 2)
        flags_unaligned_3 = evaluate_ds_inc_flags(1, 10, 0x103, 2)
        flags_aligned = evaluate_ds_inc_flags(1, 10, 0x104, 2)

        self.assertTrue(flags_unaligned_1.unaligned_addr)
        self.assertTrue(flags_unaligned_2.unaligned_addr)
        self.assertTrue(flags_unaligned_3.unaligned_addr)
        self.assertFalse(flags_aligned.unaligned_addr)

    def test_carry_generation_on_max_u32_adder(self) -> None:
        # Adder stage generates carry when incrementing 0xFFFFFFFF
        flags = evaluate_ds_inc_flags(
            old_val=0xFFFF_FFFF,
            wrap_limit=0xFFFF_FFFF,
            eff_addr=0x600,
            new_mem=0,
        )
        self.assertTrue(flags.carry)
        self.assertTrue(flags.wrapped)


# ==============================================================================
# Phase 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

# IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32 = 0x0000_0000
NEG_ZERO_F32 = 0x8000_0000
POS_INF_F32 = 0x7F80_0000
NEG_INF_F32 = 0xFF80_0000
QNAN_F32 = 0x7FC0_0000
SNAN_F32 = 0x7F80_0001
MIN_NORMAL_F32 = 0x0080_0000
MIN_SUBNORMAL_F32 = 0x0000_0001
MAX_SUBNORMAL_F32 = 0x007F_FFFF
MAX_NORMAL_F32 = 0x7F7F_FFFF
ONE_F32 = 0x3F80_0000


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK_32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    return (bits & 0x7F80_0000) == 0x7F80_0000 and (bits & 0x007F_FFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 subnormal float."""
    return (bits & 0x7F80_0000) == 0 and (bits & 0x007F_FFFF) != 0


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_increment_no_flush_to_zero(self) -> None:
        # Subnormals are not flushed to zero by LDS bitwise integer atomics
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x100
        state.lds.write_dword(addr, MIN_SUBNORMAL_F32)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 0xFFFF_FFFF)

        self.assertTrue(is_f32_subnormal(MIN_SUBNORMAL_F32))

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, MIN_SUBNORMAL_F32)
        self.assertEqual(mem, MIN_SUBNORMAL_F32 + 1)
        self.assertNotEqual(mem, 0)
        self.assertTrue(is_f32_subnormal(mem))
        self.assertEqual(state.get_vgpr(0), MIN_SUBNORMAL_F32)

    def test_subnormal_to_normal_boundary_crossing(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x120
        state.lds.write_dword(addr, MAX_SUBNORMAL_F32)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 0xFFFF_FFFF)

        self.assertTrue(is_f32_subnormal(MAX_SUBNORMAL_F32))

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, MAX_SUBNORMAL_F32)
        self.assertEqual(mem, MIN_NORMAL_F32)
        self.assertFalse(is_f32_subnormal(mem))
        self.assertEqual(state.get_vgpr(0), MAX_SUBNORMAL_F32)

    def test_signaling_nan_preserved_without_quietening(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x140
        state.lds.write_dword(addr, SNAN_F32)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 0xFFFF_FFFF)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, SNAN_F32)
        self.assertEqual(mem, SNAN_F32 + 1)
        # Verify quiet bit 22 is NOT forced high
        self.assertEqual(mem & 0x0040_0000, 0)
        self.assertTrue(is_f32_nan(mem))
        self.assertEqual(state.get_vgpr(0), SNAN_F32)

    def test_quiet_nan_wrap_boundary(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        addr = 0x160
        state.lds.write_dword(addr, QNAN_F32)
        state.set_vgpr(1, addr)
        # Limit matches QNAN_F32 -> should wrap to 0!
        state.set_vgpr(2, QNAN_F32)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        old, mem = interp.step(instr, state)
        self.assertEqual(old, QNAN_F32)
        self.assertEqual(mem, 0)
        self.assertEqual(state.get_vgpr(0), QNAN_F32)
        self.assertEqual(state.lds.read_dword(addr), 0)

    def test_positive_and_negative_zero_asymmetry(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        # POS_ZERO_F32 is integer 0
        state.lds.write_dword(0x200, POS_ZERO_F32)
        state.set_vgpr(1, 0x200)
        state.set_vgpr(2, 100)
        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )
        old, mem = interp.step(instr, state)
        self.assertEqual(old, 0)
        self.assertEqual(mem, 1)

        # NEG_ZERO_F32 is integer 0x80000000 (2^31)
        state.lds.write_dword(0x204, NEG_ZERO_F32)
        state.set_vgpr(1, 0x204)
        state.set_vgpr(2, 0xFFFF_FFFF)
        old2, mem2 = interp.step(instr, state)
        self.assertEqual(old2, 0x8000_0000)
        self.assertEqual(mem2, 0x8000_0001)

    def test_infinities_bit_patterns(self) -> None:
        state = ScalarMachineState(lds_capacity_dwords=1024)
        interp = ScalarInterpreter()

        # POS_INF_F32 is 0x7F800000
        addr = 0x300
        state.lds.write_dword(addr, POS_INF_F32)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 0xFFFF_FFFF)
        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )
        old, mem = interp.step(instr, state)
        self.assertEqual(old, POS_INF_F32)
        self.assertEqual(mem, POS_INF_F32 + 1)

        # NEG_INF_F32 is 0xFF800000
        addr2 = 0x304
        state.lds.write_dword(addr2, NEG_INF_F32)
        state.set_vgpr(1, addr2)
        state.set_vgpr(2, NEG_INF_F32) # equal to threshold -> wrap to 0
        old2, mem2 = interp.step(instr, state)
        self.assertEqual(old2, NEG_INF_F32)
        self.assertEqual(mem2, 0)


# ==============================================================================
# Phase 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Phase 8: Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_malformed_tokens(self) -> None:
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("")
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_inc_rtn_u32")
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_inc_rtn_u32 v0")
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_inc_rtn_u32 v0, v1")
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_add_rtn_u32 v0, v1, v2")
        with self.assertRaises(ValueError):
            DSIncRtnInstruction.parse("ds_inc_rtn_u32 v0, v1, v2 invalid_modifier:10")

    def test_syntax_rejection_register_classes(self) -> None:
        s0 = RegisterOperand(0, RegType.SGPR)
        v0 = RegisterOperand(0, RegType.VGPR)
        v1 = RegisterOperand(1, RegType.VGPR)

        # vdst cannot be SGPR
        with self.assertRaises(TypeError):
            DSIncRtnInstruction(vdst=s0, vaddr=v0, vdata=v1)
        # vaddr cannot be SGPR
        with self.assertRaises(TypeError):
            DSIncRtnInstruction(vdst=v0, vaddr=s0, vdata=v1)
        # vdata cannot be SGPR
        with self.assertRaises(TypeError):
            DSIncRtnInstruction(vdst=v0, vaddr=v1, vdata=s0)

    def test_syntax_rejection_register_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

    def test_syntax_rejection_offset_bounds(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            DSIncRtnInstruction(vdst=v0, vaddr=v0, vdata=v0, offset=-1)
        with self.assertRaises(ValueError):
            DSIncRtnInstruction(vdst=v0, vaddr=v0, vdata=v0, offset=0x1_0000)

    def test_syntax_rejection_lds_capacity(self) -> None:
        with self.assertRaises(ValueError):
            LDSModel(capacity_dwords=0)
        with self.assertRaises(ValueError):
            LDSModel(capacity_dwords=-16)
        with self.assertRaises(ValueError):
            LDSModel(capacity_dwords=1000)  # Not a power of two

    def test_syntax_rejection_simd_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_mutation_detection_wrap_strict_greater(self) -> None:
        # Mutant: wraps only on `old_val > wrap_limit` instead of `>=`
        def mutant_inc(old: int, limit: int) -> Tuple[int, int]:
            u_old = u32(old)
            u_lim = u32(limit)
            return u_old, 0 if u_old > u_lim else u32(u_old + 1)

        val, limit = 5, 5
        correct_old, correct_mem = oracle_ds_inc_rtn_u32(val, limit)
        mutant_old, mutant_mem = mutant_inc(val, limit)

        self.assertEqual(correct_mem, 0)
        self.assertEqual(mutant_mem, 6)
        self.assertNotEqual(correct_mem, mutant_mem)

    def test_mutation_detection_return_new_val(self) -> None:
        # Mutant: returns new_mem in vdst instead of old_val
        def mutant_inc_rtn(old: int, limit: int) -> Tuple[int, int]:
            _, new_mem = oracle_ds_inc_rtn_u32(old, limit)
            return new_mem, new_mem  # Bug: returned new_mem instead of old

        val, limit = 10, 20
        c_old, _ = oracle_ds_inc_rtn_u32(val, limit)
        m_old, _ = mutant_inc_rtn(val, limit)
        self.assertEqual(c_old, 10)
        self.assertEqual(m_old, 11)
        self.assertNotEqual(c_old, m_old)

    def test_mutation_detection_missing_offset_masking(self) -> None:
        # Mutant: fails to mask 16-bit offset
        def mutant_eff_addr(vaddr: int, offset: int) -> int:
            return (vaddr + offset) & MASK_32  # Missing `offset & 0xFFFF`

        vaddr = 0x1000
        unmasked_offset = 0x1_0020
        correct_eff = calc_lds_effective_address(vaddr, unmasked_offset)
        mutant_eff = mutant_eff_addr(vaddr, unmasked_offset)

        self.assertEqual(correct_eff, 0x1020)
        self.assertEqual(mutant_eff, 0x11020)
        self.assertNotEqual(correct_eff, mutant_eff)

    def test_mutation_detection_missing_dword_alignment(self) -> None:
        # Mutant: byte index directly instead of shifting right by 2
        def mutant_dword_idx(eff_addr: int, capacity: int) -> int:
            return eff_addr & (capacity - 1)  # Missing `>> 2`

        eff_addr = 0x103  # byte address 259
        correct_idx = calc_lds_dword_index(eff_addr, 1024)
        mutant_idx = mutant_dword_idx(eff_addr, 1024)

        self.assertEqual(correct_idx, 0x40)  # 64
        self.assertEqual(mutant_idx, 0x103)  # 259
        self.assertNotEqual(correct_idx, mutant_idx)

    def test_mutation_detection_exec_mask_leak(self) -> None:
        # Mutant: inactive lane erroneously updates LDS memory
        state = SIMDState(num_lanes=32, exec_mask=0x1, lds_capacity_dwords=1024)  # Lane 0 only active
        interp = SIMDInterpreter()

        addr_active = 0x10
        addr_inactive = 0x20
        state.lds.write_dword(addr_active, 10)
        state.lds.write_dword(addr_inactive, 50)

        # Lane 0 targets addr_active
        state.set_lane_vgpr(0, 1, addr_active)
        state.set_lane_vgpr(0, 2, 100)
        # Lane 1 targets addr_inactive
        state.set_lane_vgpr(1, 1, addr_inactive)
        state.set_lane_vgpr(1, 2, 100)

        instr = DSIncRtnInstruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata=RegisterOperand(2, RegType.VGPR),
        )

        results = interp.execute(instr, state)
        self.assertEqual(results[0], (10, 11))
        self.assertIsNone(results[1])
        # Inactive lane must not have touched LDS memory
        self.assertEqual(state.lds.read_dword(addr_inactive), 50)


if __name__ == "__main__":
    unittest.main()
