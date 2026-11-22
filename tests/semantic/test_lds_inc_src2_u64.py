from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_inc_src2_u64.

RDNA2 / GFX10 Local Data Share (LDS) 64-bit Dual-Source Atomic Increment:
Performs a 64-bit atomic wrapping increment on LDS memory location A using
the secondary LDS memory location B as the wrap threshold.

Syntax:
    ds_inc_src2_u64 vaddr [offset0:offset8] [offset1:offset8] [gds]

Semantics:
    byte_addr_a = (vaddr + (offset0 & 0xFF) * 8) & 0xFFFFFFFF
    byte_addr_b = (vaddr + (offset1 & 0xFF) * 8) & 0xFFFFFFFF
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    val_a = LDS64[qword_idx_a]
    val_b = LDS64[qword_idx_b]
    LDS64[qword_idx_a] = 0 if val_a >= val_b else (val_a + 1) & 0xFFFFFFFFFFFFFFFF

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing:
      Base byte address provided in VGPR vaddr.
      offset0 and offset1 are 8-bit unsigned immediates multiplied by 8 bytes (qword stride):
        byte_addr = (vaddr + offset * 8) mod 2^32.
      Addresses are qword-aligned: low 3 bits are dropped (byte_addr >> 3).
      Qword indices wrap modulo the allocated power-of-two capacity.
  - Wrap Condition:
      Unsigned 64-bit integer comparison between current memory value at A (val_a)
      and threshold value at B (val_b).
      When val_a >= val_b, memory location A wraps to 0.
      Otherwise, memory location A is incremented by 1 mod 2^64.
  - Non-returning:
      Vector general-purpose registers (VGPRs) are not modified.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
import unittest
from typing import Optional, Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF  # 64-bit unsigned mask
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
OFFSET8_MASK: int = 0xFF            # 8-bit immediate byte offset mask
DEFAULT_LDS_QWORDS: int = 8192      # 64 KiB capacity in 64-bit qwords (8192 * 8 bytes)
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256        # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106        # Standard GFX10 scalar register file size
DEFAULT_NUM_LANES: int = 64         # Wave64 default width


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def calc_lds_src2_u64_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_qwords: int = DEFAULT_LDS_QWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate byte addresses and qword indices for location A and location B.

    Address A = (vaddr + (offset0 & 0xFF) * 8) & MASK32
    Address B = (vaddr + (offset1 & 0xFF) * 8) & MASK32
    Qword index = (addr >> 3) & (lds_qwords - 1)

    Returns:
        Tuple[int, int, int, int]: (byte_addr_a, byte_addr_b, qword_idx_a, qword_idx_b)
    """
    byte_addr_a = u32(vaddr + ((offset0 & OFFSET8_MASK) * 8))
    byte_addr_b = u32(vaddr + ((offset1 & OFFSET8_MASK) * 8))
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    return byte_addr_a, byte_addr_b, qword_idx_a, qword_idx_b


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_inc_src2_u64(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_inc_src2_u64.

    If current memory value at A (val_a) is greater than or equal to wrap
    threshold at B (val_b), reset to 0. Otherwise, increment by 1 mod 2^64.
    """
    a = u64(val_a)
    b = u64(val_b)
    if a >= b:
        return 0
    return u64(a + 1)


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad with 64-bit access."""

    def __init__(self, capacity_qwords: int = DEFAULT_LDS_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a positive power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory = [0] * capacity_qwords

    def read_qword(self, qword_idx: int) -> int:
        """Read a 64-bit qword given an aligned qword index."""
        idx = qword_idx & (self.capacity_qwords - 1)
        return self.memory[idx]

    def write_qword(self, qword_idx: int, val: int) -> None:
        """Write a 64-bit qword given an aligned qword index."""
        idx = qword_idx & (self.capacity_qwords - 1)
        self.memory[idx] = u64(val)

    def atomic_inc_src2_u64(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        """Execute atomic 64-bit dual-source increment on LDS location A.

        Returns:
            int: The new 64-bit value written to LDS location A.
        """
        _, _, idx_a, idx_b = calc_lds_src2_u64_addresses(
            vaddr=vaddr,
            offset0=offset0,
            offset1=offset1,
            lds_qwords=self.capacity_qwords,
        )
        val_a = self.read_qword(idx_a)
        val_b = self.read_qword(idx_b)
        new_a = ref_ds_inc_src2_u64(val_a, val_b)
        self.write_qword(idx_a, new_a)
        return new_a


class TestReferenceModelAndOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_normal_increment(self) -> None:
        self.assertEqual(ref_ds_inc_src2_u64(0, 10), 1)
        self.assertEqual(ref_ds_inc_src2_u64(1, 10), 2)
        self.assertEqual(ref_ds_inc_src2_u64(8, 10), 9)
        self.assertEqual(ref_ds_inc_src2_u64(9, 10), 10)

    def test_oracle_wrap_on_equal_threshold(self) -> None:
        self.assertEqual(ref_ds_inc_src2_u64(10, 10), 0)
        self.assertEqual(ref_ds_inc_src2_u64(0x1_0000_0000, 0x1_0000_0000), 0)
        self.assertEqual(ref_ds_inc_src2_u64(MASK64, MASK64), 0)

    def test_oracle_wrap_on_greater_threshold(self) -> None:
        self.assertEqual(ref_ds_inc_src2_u64(11, 10), 0)
        self.assertEqual(ref_ds_inc_src2_u64(100, 10), 0)
        self.assertEqual(ref_ds_inc_src2_u64(MASK64, 10), 0)

    def test_oracle_zero_wrap_limit(self) -> None:
        # Any unsigned 64-bit value >= 0 wraps to 0
        self.assertEqual(ref_ds_inc_src2_u64(0, 0), 0)
        self.assertEqual(ref_ds_inc_src2_u64(1, 0), 0)
        self.assertEqual(ref_ds_inc_src2_u64(0xDEADBEEF, 0), 0)

    def test_oracle_64bit_boundary_values(self) -> None:
        # Crossing 32-bit boundary into 64-bit range
        val_32bit_max = 0xFFFF_FFFF
        self.assertEqual(ref_ds_inc_src2_u64(val_32bit_max, 0x1_0000_0000), 0x1_0000_0000)

        # Max 64-bit unsigned limit
        self.assertEqual(ref_ds_inc_src2_u64(0, MASK64), 1)
        self.assertEqual(ref_ds_inc_src2_u64(MASK64 - 1, MASK64), MASK64)
        self.assertEqual(ref_ds_inc_src2_u64(MASK64, MASK64), 0)

    def test_address_calculation_and_qword_alignment(self) -> None:
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_u64_addresses(
            vaddr=0x1000,
            offset0=2,  # 2 * 8 = 16 bytes
            offset1=4,  # 4 * 8 = 32 bytes
            lds_qwords=1024,
        )
        self.assertEqual(byte_a, 0x1010)
        self.assertEqual(byte_b, 0x1020)
        self.assertEqual(idx_a, 0x1010 >> 3)
        self.assertEqual(idx_b, 0x1020 >> 3)

        # Unaligned low bits [2:0] are dropped
        _, _, idx_unaligned, _ = calc_lds_src2_u64_addresses(
            vaddr=0x1007,
            offset0=0,
            offset1=0,
            lds_qwords=1024,
        )
        self.assertEqual(idx_unaligned, 0x1000 >> 3)

    def test_address_modulo_wrapping(self) -> None:
        # Modulo 2^32 byte address wrap
        byte_a, _, _, _ = calc_lds_src2_u64_addresses(0xFFFF_FFF8, offset0=2)
        self.assertEqual(byte_a, 0x0000_0008)

    def test_self_referencing_aliasing(self) -> None:
        # When location A and location B are identical: val_a == val_b -> wraps to 0
        self.assertEqual(ref_ds_inc_src2_u64(5, 5), 0)
        self.assertEqual(ref_ds_inc_src2_u64(0, 0), 0)
        self.assertEqual(ref_ds_inc_src2_u64(0x1234_5678_9ABC, 0x1234_5678_9ABC), 0)

    def test_lds_model_atomic_inc_src2_u64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        idx_a = 10
        idx_b = 20

        # Initialize location A = 5, location B = 8
        lds.write_qword(idx_a, 5)
        lds.write_qword(idx_b, 8)

        # Increment A with limit at B (offset0=10, offset1=20 with vaddr=0)
        new_val = lds.atomic_inc_src2_u64(vaddr=0, offset0=idx_a, offset1=idx_b)
        self.assertEqual(new_val, 6)
        self.assertEqual(lds.read_qword(idx_a), 6)
        # Location B must remain untouched (read-only wrap threshold)
        self.assertEqual(lds.read_qword(idx_b), 8)

        # Advance to threshold
        lds.write_qword(idx_a, 8)
        new_val = lds.atomic_inc_src2_u64(vaddr=0, offset0=idx_a, offset1=idx_b)
        self.assertEqual(new_val, 0)
        self.assertEqual(lds.read_qword(idx_a), 0)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_INC_SRC2_U64 = "ds_inc_src2_u64"


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
class LDSAddressExpression:
    """Dual-source LDS address expression with offset0 and offset1 immediates."""
    base_reg: RegisterOperand
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if self.base_reg.reg_type != RegType.VGPR:
            raise TypeError(f"base_reg must be VGPR, got: {self.base_reg.reg_type}")
        if not (0 <= self.offset0 <= OFFSET8_MASK):
            raise ValueError(f"offset0 out of bounds [0..{OFFSET8_MASK}]: {self.offset0}")
        if not (0 <= self.offset1 <= OFFSET8_MASK):
            raise ValueError(f"offset1 out of bounds [0..{OFFSET8_MASK}]: {self.offset1}")


@dataclass(frozen=True)
class DSIncSrc2U64Instruction:
    """Symbolic AST node for ds_inc_src2_u64 instruction."""
    addr: LDSAddressExpression
    opcode: DSOpcode = DSOpcode.DS_INC_SRC2_U64

    def format_asm(self) -> str:
        parts = [self.opcode.value, self.addr.base_reg.name]
        if self.addr.offset0 != 0:
            parts.append(f"offset0:{self.addr.offset0}")
        if self.addr.offset1 != 0:
            parts.append(f"offset1:{self.addr.offset1}")
        if self.addr.gds:
            parts.append("gds")
        return " ".join(parts)

    @classmethod
    def parse(cls, asm_line: str) -> DSIncSrc2U64Instruction:
        clean = re.sub(r"//.*|/\*.*?\*/|;.*", "", asm_line).strip()
        if not clean:
            raise ValueError("Empty assembly line")

        tokens = [t.strip() for t in clean.replace(",", " ").split() if t.strip()]
        if len(tokens) < 2:
            raise ValueError(f"Too few tokens for ds_inc_src2_u64 instruction: '{clean}'")

        op_str = tokens[0].lower()
        if op_str != DSOpcode.DS_INC_SRC2_U64.value:
            raise ValueError(f"Expected opcode {DSOpcode.DS_INC_SRC2_U64.value}, got: '{tokens[0]}'")

        m = re.match(r"^v(\d+)$", tokens[1].lower())
        if not m:
            raise ValueError(f"Expected base VGPR token (e.g. v0), got: '{tokens[1]}'")
        base_reg = RegisterOperand(int(m.group(1)), RegType.VGPR)

        offset0 = 0
        offset1 = 0
        gds = False

        for extra in tokens[2:]:
            low = extra.lower()
            m0 = re.match(r"^offset0:(\d+|0x[0-9a-fA-F]+)$", low)
            m1 = re.match(r"^offset1:(\d+|0x[0-9a-fA-F]+)$", low)
            if m0:
                val = m0.group(1)
                offset0 = int(val, 16 if val.startswith("0x") else 10)
            elif m1:
                val = m1.group(1)
                offset1 = int(val, 16 if val.startswith("0x") else 10)
            elif low == "gds":
                gds = True
            else:
                raise ValueError(f"Unrecognized instruction modifier: '{extra}'")

        addr = LDSAddressExpression(base_reg=base_reg, offset0=offset0, offset1=offset1, gds=gds)
        return cls(addr=addr)


class TestSymbolicASTAndInstruction(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        addr = LDSAddressExpression(base_reg=v0, offset0=2, offset1=4)
        instr = DSIncSrc2U64Instruction(addr=addr)

        self.assertEqual(instr.opcode, DSOpcode.DS_INC_SRC2_U64)
        self.assertEqual(instr.addr.base_reg.name, "v0")
        self.assertEqual(instr.addr.offset0, 2)
        self.assertEqual(instr.addr.offset1, 4)
        self.assertFalse(instr.addr.gds)

    def test_asm_formatting(self) -> None:
        v1 = RegisterOperand(1, RegType.VGPR)
        instr_basic = DSIncSrc2U64Instruction(addr=LDSAddressExpression(base_reg=v1))
        self.assertEqual(instr_basic.format_asm(), "ds_inc_src2_u64 v1")

        instr_offsets = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(base_reg=v1, offset0=10, offset1=20)
        )
        self.assertEqual(instr_offsets.format_asm(), "ds_inc_src2_u64 v1 offset0:10 offset1:20")

    def test_asm_parsing_valid(self) -> None:
        instr1 = DSIncSrc2U64Instruction.parse("ds_inc_src2_u64 v0")
        self.assertEqual(instr1.addr.base_reg.index, 0)
        self.assertEqual(instr1.addr.offset0, 0)
        self.assertEqual(instr1.addr.offset1, 0)

        instr2 = DSIncSrc2U64Instruction.parse("ds_inc_src2_u64 v5 offset0:3 offset1:7 ; comment")
        self.assertEqual(instr2.addr.base_reg.index, 5)
        self.assertEqual(instr2.addr.offset0, 3)
        self.assertEqual(instr2.addr.offset1, 7)

        instr3 = DSIncSrc2U64Instruction.parse("ds_inc_src2_u64 v10 offset0:0x10 offset1:0x20 gds")
        self.assertEqual(instr3.addr.offset0, 16)
        self.assertEqual(instr3.addr.offset1, 32)
        self.assertTrue(instr3.addr.gds)

    def test_syntax_rejection_registers_and_offsets(self) -> None:
        s0 = RegisterOperand(0, RegType.SGPR)
        with self.assertRaises(TypeError):
            LDSAddressExpression(base_reg=s0)

        v0 = RegisterOperand(0, RegType.VGPR)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v0, offset0=-1)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v0, offset0=256)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v0, offset1=-1)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=v0, offset1=256)

        with self.assertRaises(ValueError):
            DSIncSrc2U64Instruction.parse("ds_inc_src2_u32 v0")  # wrong opcode
        with self.assertRaises(ValueError):
            DSIncSrc2U64Instruction.parse("ds_inc_src2_u64 s0")  # SGPR not allowed
        with self.assertRaises(ValueError):
            DSIncSrc2U64Instruction.parse("ds_inc_src2_u64 v0 bad_token")


if __name__ == "__main__":
    unittest.main()
