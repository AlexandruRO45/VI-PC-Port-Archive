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


if __name__ == "__main__":
    unittest.main()
