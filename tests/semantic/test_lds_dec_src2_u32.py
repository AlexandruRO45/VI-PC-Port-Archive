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


# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState32:
    """Architectural state representation for single-thread scalar execution."""

    def __init__(
        self,
        lds_capacity_dwords: int = DEFAULT_LDS_DWORDS,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
        num_sgprs: int = DEFAULT_NUM_SGPRS,
    ) -> None:
        self.lds = LDSModel32(capacity_dwords=lds_capacity_dwords)
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


class ScalarInterpreter32:
    """Deterministic scalar instruction interpreter executing ds_dec_src2_u32."""

    def step(
        self,
        instr: DSDecSrc2U32Instruction,
        state: ScalarMachineState32,
    ) -> int:
        """Execute one ds_dec_src2_u32 instruction sequentially.

        Returns:
            int: The new 32-bit value in LDS location A.
        """
        vaddr = state.get_vgpr(instr.addr.base_reg.index)
        new_val = state.lds.atomic_dec_src2_u32(
            vaddr=vaddr,
            offset0=instr.addr.offset0,
            offset1=instr.addr.offset1,
        )
        return new_val

    def parse_and_step(
        self,
        asm_line: str,
        state: ScalarMachineState32,
    ) -> int:
        """Parse assembly text and execute one step."""
        instr = DSDecSrc2U32Instruction.parse(asm_line)
        return self.step(instr, state)


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def test_single_step_execution(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.lds.write_dword(4, 10)
        state.lds.write_dword(8, 20)
        state.set_vgpr(1, 0)

        instr = DSDecSrc2U32Instruction.parse("ds_dec_src2_u32 v1 offset0:4 offset1:8")
        new_a = interp.step(instr, state)

        self.assertEqual(new_a, 9)
        self.assertEqual(state.lds.read_dword(4), 9)
        self.assertEqual(state.lds.read_dword(8), 20)
        self.assertEqual(state.get_vgpr(1), 0)

    def test_scalar_wrapping_cycle(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        idx_a = 2
        idx_b = 3
        state.lds.write_dword(idx_a, 3)
        state.lds.write_dword(idx_b, 3) # limit = 3
        state.set_vgpr(0, 0)

        instr = DSDecSrc2U32Instruction.parse(f"ds_dec_src2_u32 v0 offset0:{idx_a} offset1:{idx_b}")

        # 3 -> 2
        self.assertEqual(interp.step(instr, state), 2)
        # 2 -> 1
        self.assertEqual(interp.step(instr, state), 1)
        # 1 -> 0
        self.assertEqual(interp.step(instr, state), 0)
        # 0 -> 3 (wrap!)
        self.assertEqual(interp.step(instr, state), 3)
        # 3 -> 2
        self.assertEqual(interp.step(instr, state), 2)

    def test_base_address_offset_addition(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.set_vgpr(2, 64) # base 64 bytes = dword 16
        # offset0 = 2 -> 64 + 8 = 72 (dword 18)
        # offset1 = 4 -> 64 + 16 = 80 (dword 20)
        state.lds.write_dword(18, 50)
        state.lds.write_dword(20, 100)

        interp.parse_and_step("ds_dec_src2_u32 v2 offset0:2 offset1:4", state)
        self.assertEqual(state.lds.read_dword(18), 49)
        self.assertEqual(state.lds.read_dword(20), 100)
        self.assertEqual(state.get_vgpr(2), 64)

    def test_non_returning_registers_preserved(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.set_vgpr(0, 128)
        state.set_vgpr(5, 0xCAFEBABE)
        state.set_vgpr(6, 0xDEADBEEF)

        interp.parse_and_step("ds_dec_src2_u32 v0", state)
        self.assertEqual(state.get_vgpr(0), 128)
        self.assertEqual(state.get_vgpr(5), 0xCAFEBABE)
        self.assertEqual(state.get_vgpr(6), 0xDEADBEEF)


# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDState32:
    """Architectural SIMD wavefront state for Wave32 and Wave64 execution."""

    def __init__(
        self,
        num_lanes: int = DEFAULT_NUM_LANES,
        exec_mask: int = MASK32,
        lds_capacity_dwords: int = DEFAULT_LDS_DWORDS,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got: {num_lanes}")
        self.num_lanes = num_lanes
        self.mask_limit = 0xFFFFFFFF if num_lanes == 32 else MASK64
        self.exec_mask = exec_mask & self.mask_limit
        self.num_vgprs = num_vgprs
        self.lds = LDSModel32(capacity_dwords=lds_capacity_dwords)
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


class SIMDInterpreter32:
    """Multi-lane SIMD interpreter executing ds_dec_src2_u32 across wavefronts."""

    def execute(
        self,
        instr: DSDecSrc2U32Instruction,
        state: SIMDState32,
    ) -> list[Optional[int]]:
        """Execute ds_dec_src2_u32 across all active lanes with LDS arbitration."""
        results: list[Optional[int]] = [None] * state.num_lanes

        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue

            vaddr = state.get_lane_vgpr(lane, instr.addr.base_reg.index)
            new_val = state.lds.atomic_dec_src2_u32(
                vaddr=vaddr,
                offset0=instr.addr.offset0,
                offset1=instr.addr.offset1,
            )
            results[lane] = new_val

        return results


class TestSIMDMultiLaneAndExecMasking(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self) -> None:
        state = SIMDState32(num_lanes=32, exec_mask=0xFFFFFFFF, lds_capacity_dwords=512)
        interp = SIMDInterpreter32()

        for lane in range(32):
            d_a = lane * 2
            d_b = lane * 2 + 1
            state.lds.write_dword(d_a, 20)
            state.lds.write_dword(d_b, 50)
            state.set_lane_vgpr(lane, 0, d_a * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        for lane in range(32):
            d_a = lane * 2
            self.assertEqual(results[lane], 19)
            self.assertEqual(state.lds.read_dword(d_a), 19)

    def test_wave64_divergent_even_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState32(num_lanes=64, exec_mask=even_mask, lds_capacity_dwords=512)
        interp = SIMDInterpreter32()

        for lane in range(64):
            d_a = lane * 2
            d_b = lane * 2 + 1
            state.lds.write_dword(d_a, 50)
            state.lds.write_dword(d_b, 100)
            state.set_lane_vgpr(lane, 0, d_a * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            d_a = lane * 2
            if lane % 2 == 0:
                self.assertEqual(results[lane], 49)
                self.assertEqual(state.lds.read_dword(d_a), 49)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lds.read_dword(d_a), 50)

    def test_zero_exec_mask(self) -> None:
        state = SIMDState32(num_lanes=64, exec_mask=0, lds_capacity_dwords=512)
        interp = SIMDInterpreter32()

        state.lds.write_dword(0, 10)
        state.lds.write_dword(1, 20)
        for lane in range(64):
            state.set_lane_vgpr(lane, 0, 0)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        for lane in range(64):
            self.assertIsNone(results[lane])
        self.assertEqual(state.lds.read_dword(0), 10)

    def test_single_lane_63_active(self) -> None:
        state = SIMDState32(num_lanes=64, exec_mask=(1 << 63), lds_capacity_dwords=512)
        interp = SIMDInterpreter32()

        state.lds.write_dword(10, 42)
        state.lds.write_dword(11, 100)
        state.set_lane_vgpr(63, 0, 10 * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        for lane in range(63):
            self.assertIsNone(results[lane])
        self.assertEqual(results[63], 41)
        self.assertEqual(state.lds.read_dword(10), 41)

    def test_multi_lane_lds_address_collision_serialization(self) -> None:
        state = SIMDState32(num_lanes=32, exec_mask=0xF, lds_capacity_dwords=512)
        interp = SIMDInterpreter32()

        d_a = 5
        d_b = 6
        state.lds.write_dword(d_a, 10)
        state.lds.write_dword(d_b, 100)
        for lane in range(4):
            state.set_lane_vgpr(lane, 0, d_a * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        # Serialized decrements: 9, 8, 7, 6
        self.assertEqual(results[0], 9)
        self.assertEqual(results[1], 8)
        self.assertEqual(results[2], 7)
        self.assertEqual(results[3], 6)
        self.assertEqual(state.lds.read_dword(d_a), 6)

    def test_multi_lane_collision_with_wrapping(self) -> None:
        state = SIMDState32(num_lanes=32, exec_mask=0xF, lds_capacity_dwords=512)
        interp = SIMDInterpreter32()

        d_a = 8
        d_b = 9
        state.lds.write_dword(d_a, 1)
        state.lds.write_dword(d_b, 3) # limit 3: 1 -> 0 -> wrap to 3 -> 2
        for lane in range(4):
            state.set_lane_vgpr(lane, 0, d_a * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        results = interp.execute(instr, state)

        self.assertEqual(results[0], 0)
        self.assertEqual(results[1], 3) # wrapped to limit 3!
        self.assertEqual(results[2], 2)
        self.assertEqual(results[3], 1)
        self.assertEqual(state.lds.read_dword(d_a), 1)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndPreservation(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_self_referencing_identical_offsets(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.lds.write_dword(10, 3)
        state.set_vgpr(0, 0)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=10,
                offset1=10,
            )
        )

        # Step 1: 3 -> 2
        self.assertEqual(interp.step(instr, state), 2)
        self.assertEqual(state.lds.read_dword(10), 2)

        # Step 2: 2 -> 1
        self.assertEqual(interp.step(instr, state), 1)
        self.assertEqual(state.lds.read_dword(10), 1)

        # Step 3: 1 -> 0
        self.assertEqual(interp.step(instr, state), 0)
        self.assertEqual(state.lds.read_dword(10), 0)

        # Step 4: 0 -> wrap to b (b=0) -> 0
        self.assertEqual(interp.step(instr, state), 0)
        self.assertEqual(state.lds.read_dword(10), 0)

        # Register v0 preserved
        self.assertEqual(state.get_vgpr(0), 0)

    def test_secondary_location_b_preserved_when_distinct(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.lds.write_dword(20, 2)
        state.lds.write_dword(21, 5)
        state.set_vgpr(1, 0)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(1, RegType.VGPR),
                offset0=20,
                offset1=21,
            )
        )

        for expected in [1, 0, 5, 4]:
            res = interp.step(instr, state)
            self.assertEqual(res, expected)
            self.assertEqual(state.lds.read_dword(20), expected)
            self.assertEqual(state.lds.read_dword(21), 5)

    def test_vaddr_register_preservation_under_scalar_step(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.set_vgpr(5, 0x200)
        state.lds.write_dword(0x200 >> 2, 100)
        state.lds.write_dword((0x200 >> 2) + 1, 200)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(5, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        interp.step(instr, state)
        self.assertEqual(state.get_vgpr(5), 0x200)

    def test_simd_register_preservation_across_all_divergent_lanes(self) -> None:
        even_mask = 0x5555555555555555
        state = SIMDState32(num_lanes=64, exec_mask=even_mask, lds_capacity_dwords=512)
        interp = SIMDInterpreter32()

        sentinels = [1000 + lane for lane in range(64)]
        for lane in range(64):
            state.set_lane_vgpr(lane, 0, sentinels[lane])
            d_a = (sentinels[lane] >> 2) & 511
            state.lds.write_dword(d_a, 10)
            state.lds.write_dword((d_a + 1) & 511, 20)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        interp.execute(instr, state)

        for lane in range(64):
            self.assertEqual(state.get_lane_vgpr(lane, 0), sentinels[lane])


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class AtomicDecSrc2U32Flags:
    """Architectural condition codes and status flags for ds_dec_src2_u32."""
    wrapped: bool
    zero: bool
    negative: bool
    borrow: bool
    carry: bool
    threshold_equal: bool
    unaligned_addr_a: bool
    unaligned_addr_b: bool


def evaluate_ds_dec_src2_u32_flags(
    val_a: int,
    val_b: int,
    eff_addr_a: int,
    eff_addr_b: int,
) -> AtomicDecSrc2U32Flags:
    """Evaluate architectural flags for 32-bit dual-source atomic decrement with wrap."""
    u_a = u32(val_a)
    u_b = u32(val_b)

    wrapped = (u_a == 0 or u_a > u_b)
    new_a = u_b if wrapped else u32(u_a - 1)
    zero = (new_a == 0)
    negative = bool(new_a & 0x8000_0000)
    # Borrow generated by 32-bit subtractor when u_a is 0 (unwrapped decrement rolls under)
    borrow = (u_a == 0)
    carry = not borrow
    threshold_equal = (u_a == u_b)

    unaligned_a = bool(eff_addr_a & 0x3)
    unaligned_b = bool(eff_addr_b & 0x3)

    return AtomicDecSrc2U32Flags(
        wrapped=wrapped,
        zero=zero,
        negative=negative,
        borrow=borrow,
        carry=carry,
        threshold_equal=threshold_equal,
        unaligned_addr_a=unaligned_a,
        unaligned_addr_b=unaligned_b,
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_normal_decrement_flags(self) -> None:
        flags = evaluate_ds_dec_src2_u32_flags(
            val_a=10,
            val_b=20,
            eff_addr_a=0x100,
            eff_addr_b=0x104,
        )
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.borrow)
        self.assertTrue(flags.carry)
        self.assertFalse(flags.threshold_equal)
        self.assertFalse(flags.unaligned_addr_a)
        self.assertFalse(flags.unaligned_addr_b)

    def test_wrapped_on_zero_flags(self) -> None:
        flags = evaluate_ds_dec_src2_u32_flags(
            val_a=0,
            val_b=50,
            eff_addr_a=0x200,
            eff_addr_b=0x204,
        )
        self.assertTrue(flags.wrapped)
        self.assertFalse(flags.zero) # new_a is 50
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.carry)

    def test_wrapped_on_greater_flags(self) -> None:
        flags = evaluate_ds_dec_src2_u32_flags(
            val_a=100,
            val_b=50,
            eff_addr_a=0x300,
            eff_addr_b=0x304,
        )
        self.assertTrue(flags.wrapped)
        self.assertFalse(flags.threshold_equal)

    def test_zero_flag_when_new_value_zero(self) -> None:
        flags = evaluate_ds_dec_src2_u32_flags(
            val_a=1,
            val_b=50,
            eff_addr_a=0x400,
            eff_addr_b=0x404,
        )
        self.assertFalse(flags.wrapped)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)

    def test_dword_unaligned_address_detection(self) -> None:
        for offset in (1, 2, 3):
            flags_a = evaluate_ds_dec_src2_u32_flags(5, 10, 0x100 + offset, 0x200)
            self.assertTrue(flags_a.unaligned_addr_a)
            self.assertFalse(flags_a.unaligned_addr_b)

            flags_b = evaluate_ds_dec_src2_u32_flags(5, 10, 0x100, 0x200 + offset)
            self.assertFalse(flags_b.unaligned_addr_a)
            self.assertTrue(flags_b.unaligned_addr_b)

        flags_aligned = evaluate_ds_dec_src2_u32_flags(5, 10, 0x100, 0x104)
        self.assertFalse(flags_aligned.unaligned_addr_a)
        self.assertFalse(flags_aligned.unaligned_addr_b)


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
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 single-precision NaN."""
    return (bits & 0x7F80_0000) == 0x7F80_0000 and (bits & 0x007F_FFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 single-precision subnormal float."""
    return (bits & 0x7F80_0000) == 0 and (bits & 0x007F_FFFF) != 0


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_decrement_no_flush_to_zero(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.lds.write_dword(10, MIN_SUBNORMAL_F32 + 1)
        state.lds.write_dword(11, MASK32)
        state.set_vgpr(0, 10 * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, MIN_SUBNORMAL_F32)
        self.assertTrue(is_f32_subnormal(new_val))
        self.assertEqual(state.lds.read_dword(10), MIN_SUBNORMAL_F32)

    def test_normal_to_subnormal_boundary_crossing(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.lds.write_dword(12, MIN_NORMAL_F32)
        state.lds.write_dword(13, MASK32)
        state.set_vgpr(0, 12 * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, MAX_SUBNORMAL_F32)
        self.assertTrue(is_f32_subnormal(new_val))
        self.assertEqual(state.lds.read_dword(12), MAX_SUBNORMAL_F32)

    def test_signaling_nan_preserved_without_quietening(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.lds.write_dword(14, SNAN_F32 + 1)
        state.lds.write_dword(15, MASK32)
        state.set_vgpr(0, 14 * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, SNAN_F32)
        # Quiet bit 22 is bit 0x0040_0000; must not be forced high
        self.assertEqual(new_val & 0x0040_0000, 0)
        self.assertTrue(is_f32_nan(new_val))

    def test_quiet_nan_wrap_boundary(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        state.lds.write_dword(16, 0)
        state.lds.write_dword(17, QNAN_F32) # 0 wraps to QNAN_F32
        state.set_vgpr(0, 16 * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, QNAN_F32)
        self.assertEqual(state.lds.read_dword(16), QNAN_F32)

    def test_single_precision_infinities(self) -> None:
        state = ScalarMachineState32(lds_capacity_dwords=512)
        interp = ScalarInterpreter32()

        # POS_INF_F32
        state.lds.write_dword(20, POS_INF_F32)
        state.lds.write_dword(21, MASK32)
        state.set_vgpr(0, 20 * 4)

        instr = DSDecSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(0, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        new_val = interp.step(instr, state)
        self.assertEqual(new_val, POS_INF_F32 - 1)

    def test_integer_32bit_boundary_extremes(self) -> None:
        extremes = [
            0x0000_0000,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FFFE,
            0xFFFF_FFFF,
        ]
        for val in extremes:
            self.assertEqual(ref_ds_dec_src2_u32(val, val), val if val == 0 else val - 1)



if __name__ == "__main__":
    unittest.main()
