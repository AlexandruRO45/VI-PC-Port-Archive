from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_dec_src2_u32.

RDNA2 / GFX10 Local Data Share (LDS) 32-bit Dual-Source Atomic Decrement:
Performs a 32-bit atomic wrapping decrement on LDS memory location A using
the secondary LDS memory location B as the wrap threshold.

Syntax:
    ds_dec_src2_u32 vaddr [offset0:offset8] [offset1:offset8] [gds]

Semantics:
    byte_addr_a = (vaddr + (offset0 & 0xFF) * 4) & 0xFFFFFFFF
    byte_addr_b = (vaddr + (offset1 & 0xFF) * 4) & 0xFFFFFFFF
    dword_idx_a = (byte_addr_a >> 2) & (lds_dwords - 1)
    dword_idx_b = (byte_addr_b >> 2) & (lds_dwords - 1)
    val_a = LDS[dword_idx_a]
    val_b = LDS[dword_idx_b]
    LDS[dword_idx_a] = val_b if (val_a == 0 or val_a > val_b) else (val_a - 1) & 0xFFFFFFFF

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing:
      Base byte address provided in VGPR vaddr.
      offset0 and offset1 are 8-bit unsigned immediates multiplied by 4 bytes (dword stride):
        byte_addr = (vaddr + offset * 4) mod 2^32.
      Addresses are dword-aligned: low 2 bits are dropped (byte_addr >> 2).
      Dword indices wrap modulo the allocated power-of-two capacity.
  - Wrap Condition:
      Unsigned 32-bit integer comparison between current memory value at A (val_a)
      and threshold value at B (val_b).
      When val_a == 0 or val_a > val_b, memory location A wraps to val_b.
      Otherwise, memory location A is decremented by 1 mod 2^32.
  - Non-returning:
      Vector general-purpose registers (VGPRs) are not modified.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import re
import struct
import unittest
from typing import Optional, Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF  # 64-bit unsigned mask
OFFSET8_MASK: int = 0xFF            # 8-bit immediate byte offset mask
DEFAULT_LDS_DWORDS: int = 16384     # 64 KiB capacity in 32-bit dwords (16384 * 4 bytes)
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256        # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106        # Standard GFX10 scalar register file size
DEFAULT_NUM_LANES: int = 64         # Wave64 default width


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def calc_lds_src2_u32_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate byte addresses and dword indices for location A and location B.

    Address A = (vaddr + (offset0 & 0xFF) * 4) & MASK32
    Address B = (vaddr + (offset1 & 0xFF) * 4) & MASK32
    Dword index = (addr >> 2) & (lds_dwords - 1)

    Returns:
        Tuple[int, int, int, int]: (byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b)
    """
    byte_addr_a = u32(vaddr + ((offset0 & OFFSET8_MASK) * 4))
    byte_addr_b = u32(vaddr + ((offset1 & OFFSET8_MASK) * 4))
    dword_idx_a = (byte_addr_a >> 2) & (lds_dwords - 1)
    dword_idx_b = (byte_addr_b >> 2) & (lds_dwords - 1)
    return byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_dec_src2_u32(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_dec_src2_u32.

    If current memory value at A (val_a) is 0 or strictly exceeds the wrap
    threshold at B (val_b), wrap to val_b. Otherwise, decrement by 1 mod 2^32.
    """
    a = u32(val_a)
    b = u32(val_b)
    if a == 0 or a > b:
        return b
    return u32(a - 1)


class LDSModel32:
    """Architectural memory model representing the Local Data Share scratchpad with 32-bit access."""

    def __init__(self, capacity_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a positive power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory = [0] * capacity_dwords

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword given an aligned dword index."""
        idx = dword_idx & (self.capacity_dwords - 1)
        return self.memory[idx]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword given an aligned dword index."""
        idx = dword_idx & (self.capacity_dwords - 1)
        self.memory[idx] = u32(val)

    def atomic_dec_src2_u32(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        """Execute atomic 32-bit dual-source decrement on LDS location A.

        Returns:
            int: The new 32-bit value written to LDS location A.
        """
        _, _, idx_a, idx_b = calc_lds_src2_u32_addresses(
            vaddr=vaddr,
            offset0=offset0,
            offset1=offset1,
            lds_dwords=self.capacity_dwords,
        )
        val_a = self.read_dword(idx_a)
        val_b = self.read_dword(idx_b)
        new_a = ref_ds_dec_src2_u32(val_a, val_b)
        self.write_dword(idx_a, new_a)
        return new_a


class TestReferenceModelAndOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_normal_decrement(self) -> None:
        self.assertEqual(ref_ds_dec_src2_u32(5, 5), 4)
        self.assertEqual(ref_ds_dec_src2_u32(4, 5), 3)
        self.assertEqual(ref_ds_dec_src2_u32(3, 5), 2)
        self.assertEqual(ref_ds_dec_src2_u32(2, 5), 1)
        self.assertEqual(ref_ds_dec_src2_u32(1, 5), 0)

    def test_oracle_wrap_on_zero(self) -> None:
        self.assertEqual(ref_ds_dec_src2_u32(0, 5), 5)
        self.assertEqual(ref_ds_dec_src2_u32(0, 100), 100)
        self.assertEqual(ref_ds_dec_src2_u32(0, MASK32), MASK32)

    def test_oracle_wrap_on_greater_threshold(self) -> None:
        self.assertEqual(ref_ds_dec_src2_u32(6, 5), 5)
        self.assertEqual(ref_ds_dec_src2_u32(100, 5), 5)
        self.assertEqual(ref_ds_dec_src2_u32(MASK32, 5), 5)

    def test_oracle_zero_wrap_limit(self) -> None:
        self.assertEqual(ref_ds_dec_src2_u32(0, 0), 0)
        self.assertEqual(ref_ds_dec_src2_u32(1, 0), 0)
        self.assertEqual(ref_ds_dec_src2_u32(10, 0), 0)

    def test_oracle_max_u32_limit(self) -> None:
        self.assertEqual(ref_ds_dec_src2_u32(MASK32, MASK32), MASK32 - 1)
        self.assertEqual(ref_ds_dec_src2_u32(1, MASK32), 0)
        self.assertEqual(ref_ds_dec_src2_u32(0, MASK32), MASK32)

    def test_address_calculation_and_dword_alignment(self) -> None:
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_u32_addresses(
            vaddr=0x1000,
            offset0=2,  # 2 * 4 = 8 bytes
            offset1=4,  # 4 * 4 = 16 bytes
            lds_dwords=1024,
        )
        self.assertEqual(byte_a, 0x1008)
        self.assertEqual(byte_b, 0x1010)
        self.assertEqual(idx_a, (0x1008 >> 2) & 1023)
        self.assertEqual(idx_b, (0x1010 >> 2) & 1023)

        # Unaligned low bits [1:0] are dropped
        _, _, idx_unaligned, _ = calc_lds_src2_u32_addresses(
            vaddr=0x1003,
            offset0=0,
            offset1=0,
            lds_dwords=1024,
        )
        self.assertEqual(idx_unaligned, (0x1000 >> 2) & 1023)

    def test_address_modulo_wrapping(self) -> None:
        byte_a, _, _, _ = calc_lds_src2_u32_addresses(0xFFFF_FFFC, offset0=2)
        self.assertEqual(byte_a, 0x0000_0004)

    def test_self_referencing_aliasing(self) -> None:
        # If val_a == 0, wraps to 0
        self.assertEqual(ref_ds_dec_src2_u32(0, 0), 0)
        # If val_a > 0 and val_a == val_b, decrements by 1
        self.assertEqual(ref_ds_dec_src2_u32(5, 5), 4)
        self.assertEqual(ref_ds_dec_src2_u32(1, 1), 0)

    def test_lds_model_atomic_dec_src2_u32(self) -> None:
        lds = LDSModel32(capacity_dwords=512)
        idx_a = 10
        idx_b = 20

        # Location A = 2, Location B = 5
        lds.write_dword(idx_a, 2)
        lds.write_dword(idx_b, 5)

        # Decrement A with limit at B
        new_val = lds.atomic_dec_src2_u32(vaddr=0, offset0=idx_a, offset1=idx_b)
        self.assertEqual(new_val, 1)
        self.assertEqual(lds.read_dword(idx_a), 1)
        self.assertEqual(lds.read_dword(idx_b), 5)

        # Decrement to 0
        new_val = lds.atomic_dec_src2_u32(vaddr=0, offset0=idx_a, offset1=idx_b)
        self.assertEqual(new_val, 0)
        self.assertEqual(lds.read_dword(idx_a), 0)

        # 0 wraps to 5!
        new_val = lds.atomic_dec_src2_u32(vaddr=0, offset0=idx_a, offset1=idx_b)
        self.assertEqual(new_val, 5)
        self.assertEqual(lds.read_dword(idx_a), 5)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_DEC_SRC2_U32 = "ds_dec_src2_u32"


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
class DSDecSrc2U32Instruction:
    """Symbolic AST node for ds_dec_src2_u32 instruction."""
    addr: LDSAddressExpression
    opcode: DSOpcode = DSOpcode.DS_DEC_SRC2_U32

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
    def parse(cls, asm_line: str) -> DSDecSrc2U32Instruction:
        clean = re.sub(r"//.*|/\*.*?\*/|;.*", "", asm_line).strip()
        if not clean:
            raise ValueError("Empty assembly line")

        tokens = [t.strip() for t in clean.replace(",", " ").split() if t.strip()]
        if len(tokens) < 2:
            raise ValueError(f"Too few tokens for ds_dec_src2_u32 instruction: '{clean}'")

        op_str = tokens[0].lower()
        if op_str != DSOpcode.DS_DEC_SRC2_U32.value:
            raise ValueError(f"Expected opcode {DSOpcode.DS_DEC_SRC2_U32.value}, got: '{tokens[0]}'")

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
        instr = DSDecSrc2U32Instruction(addr=addr)

        self.assertEqual(instr.opcode, DSOpcode.DS_DEC_SRC2_U32)
        self.assertEqual(instr.addr.base_reg.name, "v0")
        self.assertEqual(instr.addr.offset0, 2)
        self.assertEqual(instr.addr.offset1, 4)
        self.assertFalse(instr.addr.gds)

    def test_asm_formatting(self) -> None:
        v1 = RegisterOperand(1, RegType.VGPR)
        instr_basic = DSDecSrc2U32Instruction(addr=LDSAddressExpression(base_reg=v1))
        self.assertEqual(instr_basic.format_asm(), "ds_dec_src2_u32 v1")

        instr_offsets = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(base_reg=v1, offset0=10, offset1=20)
        )
        self.assertEqual(instr_offsets.format_asm(), "ds_dec_src2_u32 v1 offset0:10 offset1:20")

    def test_asm_parsing_valid(self) -> None:
        instr1 = DSDecSrc2U32Instruction.parse("ds_dec_src2_u32 v0")
        self.assertEqual(instr1.addr.base_reg.index, 0)
        self.assertEqual(instr1.addr.offset0, 0)
        self.assertEqual(instr1.addr.offset1, 0)

        instr2 = DSDecSrc2U32Instruction.parse("ds_dec_src2_u32 v5 offset0:3 offset1:7 ; comment")
        self.assertEqual(instr2.addr.base_reg.index, 5)
        self.assertEqual(instr2.addr.offset0, 3)
        self.assertEqual(instr2.addr.offset1, 7)

        instr3 = DSDecSrc2U32Instruction.parse("ds_dec_src2_u32 v10 offset0:0x10 offset1:0x20 gds")
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
            DSDecSrc2U32Instruction.parse("ds_inc_src2_u32 v0")
        with self.assertRaises(ValueError):
            DSDecSrc2U32Instruction.parse("ds_dec_src2_u32 s0")
        with self.assertRaises(ValueError):
            DSDecSrc2U32Instruction.parse("ds_dec_src2_u32 v0 bad_modifier")



if __name__ == "__main__":
    unittest.main()
