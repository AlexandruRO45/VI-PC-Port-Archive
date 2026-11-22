from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_rtn_i32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Signed 32-bit Integer Maximum with Return:
Atomically compares a signed 32-bit integer operand with the value in LDS memory,
computes the signed maximum, updates LDS with the maximum, and returns the original
32-bit value in memory to the destination VGPR.

Syntax:
    ds_max_rtn_i32 vdst, vaddr, vdata0 [offset:offset16]

Architectural Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    s_old = s32(old_val)
    s_data = s32(vdata0)
    s_max = max(s_old, s_data)
    LDS[dword_index] = u32(s_max)
    vdst = old_val

Signed 32-bit Integer Properties:
    - Range: [-2^31, 2^31 - 1] ([-2147483648, 2147483647])
    - INT32_MIN = -2147483648 (0x80000000)
    - INT32_MAX = 2147483647 (0x7FFFFFFF)
    - Signed comparison: 0 > -1 (0 > 0xFFFFFFFF), 10 > -20 (10 > 0xFFFFFFEC)
"""

import math
import re
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

INT32_MIN: int = -2147483648
INT32_MAX: int = 2147483647


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer [-2^31, 2^31 - 1]."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_max_rtn_i32(
    current_mem: int,
    data_val: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_max_rtn_i32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_val = max(s32(current_mem), s32(data_val)) & 0xFFFFFFFF

    Returns:
        (old_val, new_val)
    """
    old_val = u32(current_mem)
    s_curr = s32(current_mem)
    s_data = s32(data_val)
    s_max = max(s_curr, s_data)
    new_val = u32(s_max)
    return old_val, new_val


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.memory[idx] = u32(val)

    def atomic_max_rtn_i32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic signed 32-bit integer maximum with return on LDS.

        Returns:
            (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        curr = self.memory[idx]
        old_val, new_val = ref_ds_max_rtn_i32(curr, vdata0)
        self.memory[idx] = new_val
        return old_val, new_val


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_MAX_RTN_I32 = "ds_max_rtn_i32"


class RegType(Enum):
    """Register file types."""
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
                f"Register index out of range for {self.reg_type.name}: {self.index} (limit: {max_limit})"
            )

    def __str__(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class DSMaxRtnI32Instruction:
    """Symbolic AST node representing a ds_max_rtn_i32 instruction.

    Format:
        ds_max_rtn_i32 vdst, vaddr, vdata0 [offset:offset16]
    """
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: RegisterOperand
    offset: int = 0
    opcode: DSOpcode = DSOpcode.DS_MAX_RTN_I32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be a VGPR, got {self.vdst.reg_type}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR, got {self.vaddr.reg_type}")
        if self.vdata0.reg_type != RegType.VGPR:
            raise ValueError(f"vdata0 must be a VGPR, got {self.vdata0.reg_type}")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"offset out of range [0, 65535]: {self.offset}")

    def to_assembly(self) -> str:
        """Serialize AST node to standard assembly string."""
        asm = f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vdata0}"
        if self.offset != 0:
            asm += f" offset:{self.offset}"
        return asm

    @classmethod
    def parse(cls, asm: str) -> DSMaxRtnI32Instruction:
        """Parse an assembly string into a DSMaxRtnI32Instruction AST node."""
        cleaned = re.sub(r"[;#].*$", "", asm).strip()
        if not cleaned:
            raise ValueError("Empty assembly string")

        tokens = cleaned.split()
        if not tokens:
            raise ValueError("Empty instruction")

        op_str = tokens[0].lower()
        if op_str != DSOpcode.DS_MAX_RTN_I32.value:
            raise ValueError(f"Unsupported opcode '{tokens[0]}', expected '{DSOpcode.DS_MAX_RTN_I32.value}'")

        rest = " ".join(tokens[1:])
        parts = [p.strip() for p in rest.split(",") if p.strip()]

        if len(parts) < 3:
            raise ValueError(f"Insufficient operands for {op_str}: expected at least 3, got {len(parts)}")

        vdst_str = parts[0]
        vaddr_str = parts[1]

        # Third part might contain vdata0 and optionally offset:offset separated by space or attached
        vdata_and_mods = parts[2].split()
        vdata_str = vdata_and_mods[0]

        offset_val = 0
        mod_tokens = vdata_and_mods[1:] + parts[3:]
        for m in mod_tokens:
            m_lower = m.lower()
            if m_lower.startswith("offset:"):
                raw_offset = m[len("offset:"):]
                try:
                    offset_val = int(raw_offset, 0)
                except ValueError:
                    raise ValueError(f"Invalid integer offset value: '{raw_offset}'")
            else:
                raise ValueError(f"Unknown modifier or unexpected extra token: '{m}'")

        def parse_vgpr(reg_s: str) -> RegisterOperand:
            reg_s = reg_s.strip()
            if not (reg_s.startswith("v") or reg_s.startswith("V")):
                raise ValueError(f"Expected VGPR starting with 'v', got '{reg_s}'")
            try:
                idx = int(reg_s[1:])
            except ValueError:
                raise ValueError(f"Invalid VGPR index: '{reg_s}'")
            return RegisterOperand(idx, RegType.VGPR)

        vdst = parse_vgpr(vdst_str)
        vaddr = parse_vgpr(vaddr_str)
        vdata0 = parse_vgpr(vdata_str)

        return cls(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=offset_val)


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestLDSMaxRtnI32OracleAndModel(unittest.TestCase):
    """Phase 1: Unit tests for signed 32-bit integer oracle and LDS architectural model."""

    def test_oracle_signed_positive_vs_positive(self) -> None:
        old_val, new_val = ref_ds_max_rtn_i32(10, 20)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 20)

        old_val2, new_val2 = ref_ds_max_rtn_i32(100, 50)
        self.assertEqual(old_val2, 100)
        self.assertEqual(new_val2, 100)

    def test_oracle_signed_positive_vs_negative(self) -> None:
        # In signed: 10 > -20. In unsigned: 10 < 0xFFFFFFEC.
        old_val, new_val = ref_ds_max_rtn_i32(10, u32(-20))
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 10)

        # Negative in memory, positive in operand
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(-50), 5)
        self.assertEqual(old_val2, u32(-50))
        self.assertEqual(new_val2, 5)

    def test_oracle_signed_negative_vs_negative(self) -> None:
        # -10 > -30
        old_val, new_val = ref_ds_max_rtn_i32(u32(-30), u32(-10))
        self.assertEqual(old_val, u32(-30))
        self.assertEqual(s32(new_val), -10)

        # -5 > -100
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(-5), u32(-100))
        self.assertEqual(old_val2, u32(-5))
        self.assertEqual(s32(new_val2), -5)

    def test_oracle_zero_vs_negative(self) -> None:
        # 0 > -1 (0x00000000 > 0xFFFFFFFF in signed i32)
        old_val, new_val = ref_ds_max_rtn_i32(0, u32(-1))
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0)

        # -1 in memory, 0 in data
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(-1), 0)
        self.assertEqual(old_val2, u32(-1))
        self.assertEqual(new_val2, 0)

    def test_oracle_int32_extremes(self) -> None:
        # INT32_MIN (-2147483648) vs INT32_MAX (2147483647)
        old_val, new_val = ref_ds_max_rtn_i32(u32(INT32_MIN), u32(INT32_MAX))
        self.assertEqual(old_val, u32(INT32_MIN))
        self.assertEqual(new_val, u32(INT32_MAX))

        # INT32_MIN vs 0
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(INT32_MIN), 0)
        self.assertEqual(old_val2, u32(INT32_MIN))
        self.assertEqual(new_val2, 0)

    def test_oracle_equal_operands(self) -> None:
        old_val, new_val = ref_ds_max_rtn_i32(42, 42)
        self.assertEqual(old_val, 42)
        self.assertEqual(new_val, 42)

    def test_lds_memory_basic_operations(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        lds.write_dword(0x100, 0x12345678)
        self.assertEqual(lds.read_dword(0x100), 0x12345678)

    def test_lds_dword_alignment(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        lds.write_dword(0x10, 0xDEADBEEF)
        # Low 2 bits (bytes 0, 1, 2, 3) must access the exact same dword
        self.assertEqual(lds.read_dword(0x10), 0xDEADBEEF)
        self.assertEqual(lds.read_dword(0x11), 0xDEADBEEF)
        self.assertEqual(lds.read_dword(0x12), 0xDEADBEEF)
        self.assertEqual(lds.read_dword(0x13), 0xDEADBEEF)

    def test_lds_memory_capacity_wrapping(self) -> None:
        # 1024 dwords = 4096 bytes capacity
        lds = LDSMemory(size_dwords=1024)
        lds.write_dword(0, 0xCAFEBABE)
        self.assertEqual(lds.read_dword(4096), 0xCAFEBABE)
        self.assertEqual(lds.read_dword(8192), 0xCAFEBABE)

    def test_lds_atomic_max_rtn_i32(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        # Initialize memory with -10
        lds.write_dword(0x40, u32(-10))

        # Atomic max with 5 -> old should be -10, new memory should be 5
        old_val, new_val = lds.atomic_max_rtn_i32(0x40, 5)
        self.assertEqual(old_val, u32(-10))
        self.assertEqual(new_val, 5)
        self.assertEqual(lds.read_dword(0x40), 5)

        # Atomic max with -20 -> old should be 5, new memory should remain 5
        old_val2, new_val2 = lds.atomic_max_rtn_i32(0x40, u32(-20))
        self.assertEqual(old_val2, 5)
        self.assertEqual(new_val2, 5)
        self.assertEqual(lds.read_dword(0x40), 5)

    def test_lds_bank_distribution(self) -> None:
        # Banks cycle through 0..31 for consecutive dwords
        for i in range(64):
            expected_bank = i % 32
            eff_addr = i * 4
            self.assertEqual(calc_lds_bank(eff_addr), expected_bank)


class TestDSMaxRtnI32AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_operand_valid(self) -> None:
        vgpr = RegisterOperand(0, RegType.VGPR)
        self.assertEqual(str(vgpr), "v0")
        self.assertEqual(vgpr.index, 0)
        self.assertEqual(vgpr.reg_type, RegType.VGPR)

        vgpr_max = RegisterOperand(255, RegType.VGPR)
        self.assertEqual(str(vgpr_max), "v255")

    def test_register_operand_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

    def test_instruction_construction_and_to_assembly(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)

        inst1 = DSMaxRtnI32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0)
        self.assertEqual(inst1.to_assembly(), "ds_max_rtn_i32 v0, v1, v2")

        inst2 = DSMaxRtnI32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=128)
        self.assertEqual(inst2.to_assembly(), "ds_max_rtn_i32 v0, v1, v2 offset:128")

    def test_instruction_parsing_valid(self) -> None:
        inst1 = DSMaxRtnI32Instruction.parse("ds_max_rtn_i32 v0, v1, v2")
        self.assertEqual(inst1.vdst.index, 0)
        self.assertEqual(inst1.vaddr.index, 1)
        self.assertEqual(inst1.vdata0.index, 2)
        self.assertEqual(inst1.offset, 0)

        inst2 = DSMaxRtnI32Instruction.parse("ds_max_rtn_i32 v10, v20, v30 offset:256")
        self.assertEqual(inst2.vdst.index, 10)
        self.assertEqual(inst2.vaddr.index, 20)
        self.assertEqual(inst2.vdata0.index, 30)
        self.assertEqual(inst2.offset, 256)

        # Hex offset and case insensitivity
        inst3 = DSMaxRtnI32Instruction.parse("DS_MAX_RTN_I32 v5, v6, v7 offset:0x40")
        self.assertEqual(inst3.offset, 64)

    def test_instruction_roundtrip(self) -> None:
        vdst = RegisterOperand(4, RegType.VGPR)
        vaddr = RegisterOperand(8, RegType.VGPR)
        vdata0 = RegisterOperand(12, RegType.VGPR)
        inst = DSMaxRtnI32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=60)
        parsed = DSMaxRtnI32Instruction.parse(inst.to_assembly())
        self.assertEqual(inst, parsed)


if __name__ == "__main__":
    unittest.main()
