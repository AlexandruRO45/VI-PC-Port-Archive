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
import struct
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


# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState64:
    """Architectural state representation for single-thread scalar execution."""

    def __init__(
        self,
        lds_capacity_qwords: int = DEFAULT_LDS_QWORDS,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
        num_sgprs: int = DEFAULT_NUM_SGPRS,
    ) -> None:
        self.lds = LDSModel64(capacity_qwords=lds_capacity_qwords)
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


class ScalarInterpreter64:
    """Deterministic scalar instruction interpreter executing ds_inc_src2_u64."""

    def step(
        self,
        instr: DSIncSrc2U64Instruction,
        state: ScalarMachineState64,
    ) -> int:
        """Execute one ds_inc_src2_u64 instruction sequentially.

        Returns:
            int: The new 64-bit value in LDS location A.
        """
        vaddr = state.get_vgpr(instr.addr.base_reg.index)
        new_val = state.lds.atomic_inc_src2_u64(
            vaddr=vaddr,
            offset0=instr.addr.offset0,
            offset1=instr.addr.offset1,
        )
        # Note: ds_inc_src2_u64 is non-returning; registers are untouched.
        return new_val

    def parse_and_step(
        self,
        asm_line: str,
        state: ScalarMachineState64,
    ) -> int:
        """Parse assembly text and execute one step."""
        instr = DSIncSrc2U64Instruction.parse(asm_line)
        return self.step(instr, state)


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def test_single_step_execution(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        # Location A at qword 4 (offset0=4, 32 bytes), Location B at qword 8 (offset1=8, 64 bytes)
        state.lds.write_qword(4, 10)
        state.lds.write_qword(8, 20)
        state.set_vgpr(1, 0) # vaddr = 0

        instr = DSIncSrc2U64Instruction.parse("ds_inc_src2_u64 v1 offset0:4 offset1:8")
        new_a = interp.step(instr, state)

        self.assertEqual(new_a, 11)
        self.assertEqual(state.lds.read_qword(4), 11)
        self.assertEqual(state.lds.read_qword(8), 20)
        # Verify v1 is unmodified
        self.assertEqual(state.get_vgpr(1), 0)

    def test_scalar_wrapping_cycle(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        idx_a = 2
        idx_b = 3
        state.lds.write_qword(idx_a, 0)
        state.lds.write_qword(idx_b, 3) # wrap limit = 3
        state.set_vgpr(0, 0)

        instr = DSIncSrc2U64Instruction.parse(f"ds_inc_src2_u64 v0 offset0:{idx_a} offset1:{idx_b}")

        # Step 1: 0 -> 1
        self.assertEqual(interp.step(instr, state), 1)
        # Step 2: 1 -> 2
        self.assertEqual(interp.step(instr, state), 2)
        # Step 3: 2 -> 3
        self.assertEqual(interp.step(instr, state), 3)
        # Step 4: 3 -> 0 (wrap!)
        self.assertEqual(interp.step(instr, state), 0)
        # Step 5: 0 -> 1
        self.assertEqual(interp.step(instr, state), 1)

    def test_base_address_offset_addition(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        # Base byte address = 64 (qword 8)
        state.set_vgpr(2, 64)
        # offset0 = 2 -> 64 + 16 = 80 (qword 10)
        # offset1 = 4 -> 64 + 32 = 96 (qword 12)
        state.lds.write_qword(10, 100)
        state.lds.write_qword(12, 200)

        interp.parse_and_step("ds_inc_src2_u64 v2 offset0:2 offset1:4", state)
        self.assertEqual(state.lds.read_qword(10), 101)
        self.assertEqual(state.lds.read_qword(12), 200)
        self.assertEqual(state.get_vgpr(2), 64)

    def test_non_returning_registers_preserved(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        state.set_vgpr(0, 128)
        state.set_vgpr(5, 0xCAFEBABE)
        state.set_vgpr(6, 0xDEADBEEF)

        interp.parse_and_step("ds_inc_src2_u64 v0", state)
        self.assertEqual(state.get_vgpr(0), 128)
        self.assertEqual(state.get_vgpr(5), 0xCAFEBABE)
        self.assertEqual(state.get_vgpr(6), 0xDEADBEEF)


# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDState64:
    """Architectural SIMD wavefront state for Wave32 and Wave64 execution."""

    def __init__(
        self,
        num_lanes: int = DEFAULT_NUM_LANES,
        exec_mask: int = MASK32,
        lds_capacity_qwords: int = DEFAULT_LDS_QWORDS,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got: {num_lanes}")
        self.num_lanes = num_lanes
        self.mask_limit = 0xFFFFFFFF if num_lanes == 32 else MASK64
        self.exec_mask = exec_mask & self.mask_limit
        self.num_vgprs = num_vgprs
        self.lds = LDSModel64(capacity_qwords=lds_capacity_qwords)
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


class SIMDInterpreter64:
    """Multi-lane SIMD interpreter executing ds_inc_src2_u64 across wavefronts."""

    def execute(
        self,
        instr: DSIncSrc2U64Instruction,
        state: SIMDState64,
    ) -> list[Optional[int]]:
        """Execute ds_inc_src2_u64 across all active lanes with LDS arbitration."""
        results: list[Optional[int]] = [None] * state.num_lanes

        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue

            vaddr = state.get_lane_vgpr(lane, instr.addr.base_reg.index)
            new_val = state.lds.atomic_inc_src2_u64(
                vaddr=vaddr,
                offset0=instr.addr.offset0,
                offset1=instr.addr.offset1,
            )
            results[lane] = new_val

        return results


class TestSIMDMultiLaneAndExecMasking(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        state = SIMDState64(num_lanes=32, exec_mask=0xFFFFFFFF, lds_capacity_qwords=512)
        interp = SIMDInterpreter64()

        # Each lane has its own distinct target A and limit B
        # Lane i: A at 2*i, B at 2*i + 1
        for lane in range(32):
            q_a = lane * 2
            q_b = lane * 2 + 1
            state.lds.write_qword(q_a, lane * 5)
            state.lds.write_qword(q_b, 1000)
            state.set_lane_vgpr(lane, 0, q_a * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1, # offset1 adds 1 qword = 8 bytes
            )
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            q_a = lane * 2
            expected = lane * 5 + 1
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.lds.read_qword(q_a), expected)

    def test_wave64_divergent_even_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState64(num_lanes=64, exec_mask=even_mask, lds_capacity_qwords=512)
        interp = SIMDInterpreter64()

        for lane in range(64):
            q_a = lane * 2
            q_b = lane * 2 + 1
            state.lds.write_qword(q_a, 50)
            state.lds.write_qword(q_b, 100)
            state.set_lane_vgpr(lane, 0, q_a * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            q_a = lane * 2
            if lane % 2 == 0:
                self.assertEqual(results[lane], 51)
                self.assertEqual(state.lds.read_qword(q_a), 51)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lds.read_qword(q_a), 50)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState64(num_lanes=64, exec_mask=0, lds_capacity_qwords=512)
        interp = SIMDInterpreter64()

        state.lds.write_qword(0, 10)
        state.lds.write_qword(1, 20)
        for lane in range(64):
            state.set_lane_vgpr(lane, 0, 0)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            self.assertIsNone(results[lane])
        self.assertEqual(state.lds.read_qword(0), 10)

    def test_single_lane_63_active(self) -> None:
        state = SIMDState64(num_lanes=64, exec_mask=(1 << 63), lds_capacity_qwords=512)
        interp = SIMDInterpreter64()

        state.lds.write_qword(10, 42)
        state.lds.write_qword(11, 100)
        state.set_lane_vgpr(63, 0, 10 * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        for lane in range(63):
            self.assertIsNone(results[lane])
        self.assertEqual(results[63], 43)
        self.assertEqual(state.lds.read_qword(10), 43)

    def test_multi_lane_lds_address_collision_serialization(self) -> None:
        # Lanes 0..3 active, all colliding on location A at qword 5
        state = SIMDState64(num_lanes=32, exec_mask=0xF, lds_capacity_qwords=512)
        interp = SIMDInterpreter64()

        q_a = 5
        q_b = 6
        state.lds.write_qword(q_a, 0)
        state.lds.write_qword(q_b, 100) # limit 100
        for lane in range(4):
            state.set_lane_vgpr(lane, 0, q_a * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        # Serialized increments: 1, 2, 3, 4
        self.assertEqual(results[0], 1)
        self.assertEqual(results[1], 2)
        self.assertEqual(results[2], 3)
        self.assertEqual(results[3], 4)
        self.assertEqual(state.lds.read_qword(q_a), 4)

    def test_multi_lane_collision_with_wrapping(self) -> None:
        state = SIMDState64(num_lanes=32, exec_mask=0xF, lds_capacity_qwords=512)
        interp = SIMDInterpreter64()

        q_a = 8
        q_b = 9
        state.lds.write_qword(q_a, 0)
        state.lds.write_qword(q_b, 2) # limit 2: 0 -> 1 -> 2 -> 0 -> 1
        for lane in range(4):
            state.set_lane_vgpr(lane, 0, q_a * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        self.assertEqual(results[0], 1)
        self.assertEqual(results[1], 2)
        self.assertEqual(results[2], 0) # wrapped!
        self.assertEqual(results[3], 1)
        self.assertEqual(state.lds.read_qword(q_a), 1)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_self_referencing_identical_offsets_always_wrap_to_zero(self) -> None:
        # When offset0 == offset1, Location A and Location B are the exact same qword
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        # Location qword 10 initially has 42
        state.lds.write_qword(10, 42)
        state.set_vgpr(0, 0)

        # offset0:10, offset1:10 -> A and B both point to qword 10
        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=10,
                offset1=10,
            )
        )

        # 42 >= 42 -> wraps to 0
        new_val = interp.step(instr, state)
        self.assertEqual(new_val, 0)
        self.assertEqual(state.lds.read_qword(10), 0)

        # Second step: 0 >= 0 -> wraps to 0 again
        new_val2 = interp.step(instr, state)
        self.assertEqual(new_val2, 0)
        self.assertEqual(state.lds.read_qword(10), 0)

        # Register v0 must be intact
        self.assertEqual(state.get_vgpr(0), 0)

    def test_secondary_location_b_preserved_when_distinct(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        state.lds.write_qword(20, 1)
        state.lds.write_qword(21, 5) # limit = 5
        state.set_vgpr(1, 0)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(1, RegType.VGPR),
                offset0=20,
                offset1=21,
            )
        )

        for expected in [2, 3, 4, 5, 0]:
            res = interp.step(instr, state)
            self.assertEqual(res, expected)
            self.assertEqual(state.lds.read_qword(20), expected)
            # Location B must remain strictly 5 throughout
            self.assertEqual(state.lds.read_qword(21), 5)

    def test_vaddr_register_preservation_under_scalar_step(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        # v5 holds base address 0x200
        state.set_vgpr(5, 0x200)
        state.lds.write_qword(0x200 >> 3, 100)
        state.lds.write_qword((0x200 >> 3) + 1, 200)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(5, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        interp.step(instr, state)
        # Verify base VGPR is not clobbered
        self.assertEqual(state.get_vgpr(5), 0x200)

    def test_simd_register_preservation_across_all_divergent_lanes(self) -> None:
        # Wave64 with alternating mask
        even_mask = 0x5555555555555555
        state = SIMDState64(num_lanes=64, exec_mask=even_mask, lds_capacity_qwords=512)
        interp = SIMDInterpreter64()

        sentinels = [1000 + lane for lane in range(64)]
        for lane in range(64):
            state.set_lane_vgpr(lane, 0, sentinels[lane])
            q_a = (sentinels[lane] >> 3) & 511
            state.lds.write_qword(q_a, 10)
            state.lds.write_qword((q_a + 1) & 511, 20)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        interp.execute(instr, state)

        # Every lane's VGPR must be perfectly preserved
        for lane in range(64):
            self.assertEqual(state.get_lane_vgpr(lane, 0), sentinels[lane])


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class AtomicIncSrc2U64Flags:
    """Architectural condition codes and status flags for ds_inc_src2_u64."""
    wrapped: bool
    zero: bool
    negative: bool
    carry: bool
    borrow: bool
    threshold_equal: bool
    unaligned_addr_a: bool
    unaligned_addr_b: bool


def evaluate_ds_inc_src2_u64_flags(
    val_a: int,
    val_b: int,
    eff_addr_a: int,
    eff_addr_b: int,
) -> AtomicIncSrc2U64Flags:
    """Evaluate architectural flags for 64-bit dual-source atomic increment with wrap."""
    u_a = u64(val_a)
    u_b = u64(val_b)

    wrapped = (u_a >= u_b)
    new_a = 0 if wrapped else u64(u_a + 1)
    zero = (new_a == 0)
    negative = bool(new_a & 0x8000_0000_0000_0000)
    borrow = (u_a < u_b)
    threshold_equal = (u_a == u_b)

    # Carry out generated by the 64-bit adder stage before the wrap multiplexer
    carry = bool((u_a + 1) > MASK64)
    # Low 3 bits indicate non-qword-aligned byte address (8-byte qword boundary)
    unaligned_a = bool(eff_addr_a & 0x7)
    unaligned_b = bool(eff_addr_b & 0x7)

    return AtomicIncSrc2U64Flags(
        wrapped=wrapped,
        zero=zero,
        negative=negative,
        carry=carry,
        borrow=borrow,
        threshold_equal=threshold_equal,
        unaligned_addr_a=unaligned_a,
        unaligned_addr_b=unaligned_b,
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_normal_increment_flags(self) -> None:
        flags = evaluate_ds_inc_src2_u64_flags(
            val_a=10,
            val_b=20,
            eff_addr_a=0x100,
            eff_addr_b=0x108,
        )
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.carry)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.threshold_equal)
        self.assertFalse(flags.unaligned_addr_a)
        self.assertFalse(flags.unaligned_addr_b)

    def test_wrapped_on_equal_flags(self) -> None:
        flags = evaluate_ds_inc_src2_u64_flags(
            val_a=50,
            val_b=50,
            eff_addr_a=0x200,
            eff_addr_b=0x208,
        )
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.zero)  # wraps to 0
        self.assertFalse(flags.negative)
        self.assertTrue(flags.threshold_equal)
        self.assertFalse(flags.borrow)

    def test_negative_flag_when_msb_set(self) -> None:
        msb_minus_one = 0x8000_0000_0000_0000 - 1
        flags = evaluate_ds_inc_src2_u64_flags(
            val_a=msb_minus_one,
            val_b=MASK64,
            eff_addr_a=0x300,
            eff_addr_b=0x308,
        )
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)  # new_a has MSB set

    def test_carry_generation_on_max_u64_adder(self) -> None:
        flags = evaluate_ds_inc_src2_u64_flags(
            val_a=MASK64,
            val_b=MASK64,
            eff_addr_a=0x400,
            eff_addr_b=0x408,
        )
        self.assertTrue(flags.carry)
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.zero)

    def test_qword_unaligned_address_detection(self) -> None:
        # Qword alignment requires low 3 bits to be 0 (multiple of 8)
        for offset in range(1, 8):
            flags_a = evaluate_ds_inc_src2_u64_flags(1, 10, 0x100 + offset, 0x200)
            self.assertTrue(flags_a.unaligned_addr_a)
            self.assertFalse(flags_a.unaligned_addr_b)

            flags_b = evaluate_ds_inc_src2_u64_flags(1, 10, 0x100, 0x200 + offset)
            self.assertFalse(flags_b.unaligned_addr_a)
            self.assertTrue(flags_b.unaligned_addr_b)

        flags_aligned = evaluate_ds_inc_src2_u64_flags(1, 10, 0x100, 0x208)
        self.assertFalse(flags_aligned.unaligned_addr_a)
        self.assertFalse(flags_aligned.unaligned_addr_b)


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
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Pack IEEE 754 double-precision float into 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_f64_nan(bits: int) -> bool:
    """Test if 64-bit bit pattern is an IEEE 754 double-precision NaN."""
    return (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def is_f64_subnormal(bits: int) -> bool:
    """Test if 64-bit bit pattern is an IEEE 754 double-precision subnormal float."""
    return (bits & 0x7FF0_0000_0000_0000) == 0 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_increment_no_flush_to_zero(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        # Location A has subnormal float
        state.lds.write_qword(10, MIN_SUBNORMAL_F64)
        state.lds.write_qword(11, MASK64)
        state.set_vgpr(0, 10 * 8)

        self.assertTrue(is_f64_subnormal(MIN_SUBNORMAL_F64))

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, MIN_SUBNORMAL_F64 + 1)
        self.assertNotEqual(new_val, 0)
        self.assertTrue(is_f64_subnormal(new_val))
        self.assertEqual(state.lds.read_qword(10), MIN_SUBNORMAL_F64 + 1)

    def test_subnormal_to_normal_boundary_crossing(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        state.lds.write_qword(12, MAX_SUBNORMAL_F64)
        state.lds.write_qword(13, MASK64)
        state.set_vgpr(0, 12 * 8)

        self.assertTrue(is_f64_subnormal(MAX_SUBNORMAL_F64))

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, MIN_NORMAL_F64)
        self.assertFalse(is_f64_subnormal(new_val))
        self.assertEqual(state.lds.read_qword(12), MIN_NORMAL_F64)

    def test_signaling_nan_preserved_without_quietening(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        state.lds.write_qword(14, SNAN_F64)
        state.lds.write_qword(15, MASK64)
        state.set_vgpr(0, 14 * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, SNAN_F64 + 1)
        # Quiet bit 51 is bit 0x0008_0000_0000_0000; must not be forced high
        self.assertEqual(new_val & 0x0008_0000_0000_0000, 0)
        self.assertTrue(is_f64_nan(new_val))

    def test_quiet_nan_wrap_boundary(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        state.lds.write_qword(16, QNAN_F64)
        state.lds.write_qword(17, QNAN_F64) # Equal to threshold -> wraps to 0
        state.set_vgpr(0, 16 * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, 0)
        self.assertEqual(state.lds.read_qword(16), 0)

    def test_double_precision_infinities(self) -> None:
        state = ScalarMachineState64(lds_capacity_qwords=512)
        interp = ScalarInterpreter64()

        # POS_INF_F64
        state.lds.write_qword(20, POS_INF_F64)
        state.lds.write_qword(21, MASK64)
        state.set_vgpr(0, 20 * 8)

        instr = DSIncSrc2U64Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, POS_INF_F64 + 1)

        # NEG_INF_F64
        state.lds.write_qword(22, NEG_INF_F64)
        state.lds.write_qword(23, NEG_INF_F64) # wrap on match
        state.set_vgpr(0, 22 * 8)

        new_val2 = interp.step(instr, state)
        self.assertEqual(new_val2, 0)

    def test_integer_64bit_boundary_extremes(self) -> None:
        extremes = [
            0x0000_0000_0000_0000,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFE,
            0xFFFF_FFFF_FFFF_FFFF,
        ]
        for val in extremes:
            # Check increment below limit
            self.assertEqual(ref_ds_inc_src2_u64(val, MASK64), 0 if val == MASK64 else val + 1)
            # Check wrap at equal limit
            self.assertEqual(ref_ds_inc_src2_u64(val, val), 0)


if __name__ == "__main__":
    unittest.main()
