from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_dec_src2_u64.

RDNA2 / GFX10 Local Data Share (LDS) 64-bit Dual-Source Atomic Decrement:
Performs a 64-bit atomic wrapping decrement on LDS memory location A using
the secondary LDS memory location B as the wrap threshold.

Syntax:
    ds_dec_src2_u64 vaddr [offset0:offset8] [offset1:offset8] [gds]

Semantics:
    byte_addr_a = (vaddr + (offset0 & 0xFF) * 8) & 0xFFFFFFFF
    byte_addr_b = (vaddr + (offset1 & 0xFF) * 8) & 0xFFFFFFFF
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    val_a = LDS64[qword_idx_a]
    val_b = LDS64[qword_idx_b]
    if val_a == 0 or val_a > val_b:
        new_val_a = val_b
    else:
        new_val_a = (val_a - 1) & 0xFFFFFFFFFFFFFFFF
    LDS64[qword_idx_a] = new_val_a

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
      When val_a == 0 or val_a > val_b, memory location A resets to val_b.
      Otherwise, memory location A is decremented by 1 mod 2^64.
  - Non-returning:
      Vector general-purpose registers (VGPRs) are not modified.
"""

import struct
import unittest
from typing import Tuple

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


def ref_ds_dec_src2_u64(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_dec_src2_u64.

    If current memory value at A (val_a) is 0 or strictly greater than threshold
    at B (val_b), reset to val_b. Otherwise, decrement by 1 mod 2^64.
    """
    a = u64(val_a)
    b = u64(val_b)
    if a == 0 or a > b:
        return b
    return u64(a - 1)


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

    def atomic_dec_src2_u64(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        """Execute atomic 64-bit dual-source decrement on LDS location A.

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
        new_val_a = ref_ds_dec_src2_u64(val_a, val_b)
        self.write_qword(idx_a, new_val_a)
        return new_val_a


class TestReferenceOracleAndLDSModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_normal_decrement(self) -> None:
        # 5 decrements to 4 with threshold 10
        self.assertEqual(ref_ds_dec_src2_u64(5, 10), 4)
        self.assertEqual(ref_ds_dec_src2_u64(10, 10), 9)
        self.assertEqual(ref_ds_dec_src2_u64(1, 10), 0)

    def test_oracle_wrap_on_zero(self) -> None:
        # At 0, wraps to threshold val_b
        self.assertEqual(ref_ds_dec_src2_u64(0, 10), 10)
        self.assertEqual(ref_ds_dec_src2_u64(0, 0x1_0000_0000), 0x1_0000_0000)

    def test_oracle_wrap_on_greater_threshold(self) -> None:
        # If val_a > val_b, resets to val_b
        self.assertEqual(ref_ds_dec_src2_u64(11, 10), 10)
        self.assertEqual(ref_ds_dec_src2_u64(100, 5), 5)
        self.assertEqual(ref_ds_dec_src2_u64(MASK64, 50), 50)

    def test_oracle_zero_threshold(self) -> None:
        # Threshold 0: if val_a == 0 -> 0; if val_a > 0 -> 0
        self.assertEqual(ref_ds_dec_src2_u64(0, 0), 0)
        self.assertEqual(ref_ds_dec_src2_u64(5, 0), 0)

    def test_oracle_max_u64_boundary(self) -> None:
        # Max u64 threshold
        self.assertEqual(ref_ds_dec_src2_u64(MASK64, MASK64), MASK64 - 1)
        self.assertEqual(ref_ds_dec_src2_u64(0, MASK64), MASK64)

    def test_address_calculation_and_stride(self) -> None:
        base = 0x1000
        off0 = 2  # 2 * 8 = 16 bytes
        off1 = 5  # 5 * 8 = 40 bytes
        addr_a, addr_b, idx_a, idx_b = calc_lds_src2_u64_addresses(base, off0, off1)

        self.assertEqual(addr_a, 0x1010)
        self.assertEqual(addr_b, 0x1028)
        self.assertEqual(idx_a, 0x1010 >> 3)
        self.assertEqual(idx_b, 0x1028 >> 3)

    def test_offset8_masking(self) -> None:
        # Offsets are 8-bit unsigned
        base = 0x2000
        off = 0x102  # Masked to 2 -> 16 bytes
        addr_a, _, idx_a, _ = calc_lds_src2_u64_addresses(base, off, 0)
        self.assertEqual(addr_a, 0x2010)
        self.assertEqual(idx_a, 0x2010 >> 3)

    def test_lds_model_atomic_dec_src2_u64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        vaddr = 0x100

        # Location A: vaddr + 0*8 = idx 32
        # Location B: vaddr + 1*8 = idx 33
        idx_a = 0x100 >> 3
        idx_b = (0x100 + 8) >> 3

        lds.write_qword(idx_a, 5)
        lds.write_qword(idx_b, 10)  # threshold 10

        new_val = lds.atomic_dec_src2_u64(vaddr, offset0=0, offset1=1)
        self.assertEqual(new_val, 4)
        self.assertEqual(lds.read_qword(idx_a), 4)
        self.assertEqual(lds.read_qword(idx_b), 10)  # Threshold untouched!

        # Decrement down to 0
        lds.write_qword(idx_a, 1)
        new_val = lds.atomic_dec_src2_u64(vaddr, offset0=0, offset1=1)
        self.assertEqual(new_val, 0)

        # Wrap at 0
        new_val = lds.atomic_dec_src2_u64(vaddr, offset0=0, offset1=1)
        self.assertEqual(new_val, 10)
        self.assertEqual(lds.read_qword(idx_a), 10)


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


class DsSrc2Opcode(enum.Enum):
    """LDS dual-source instruction opcodes."""
    DS_DEC_SRC2_U64 = "ds_dec_src2_u64"
    DS_INC_SRC2_U64 = "ds_inc_src2_u64"


@dataclasses.dataclass(frozen=True)
class DsSrc2Instruction:
    """Symbolic AST node representing a dual-source DS atomic instruction."""
    opcode: DsSrc2Opcode
    vaddr: Register
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR: {self.vaddr}")
        if self.offset0 < 0 or self.offset0 > 255:
            raise ValueError(f"offset0 must be 8-bit unsigned [0, 255], got {self.offset0}")
        if self.offset1 < 0 or self.offset1 > 255:
            raise ValueError(f"offset1 must be 8-bit unsigned [0, 255], got {self.offset1}")

    def to_assembly(self) -> str:
        parts: list[str] = [self.opcode.value, self.vaddr.format_asm()]
        if self.offset0 != 0:
            parts.append(f"offset0:{self.offset0}")
        if self.offset1 != 0:
            parts.append(f"offset1:{self.offset1}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)


class DsDecSrc2Expr:
    """Typed symbolic expression for ds_dec_src2_u64."""

    def __init__(self, vaddr: int, offset0: int = 0, offset1: int = 0) -> None:
        self.vaddr = vaddr & MASK32
        self.offset0 = offset0 & OFFSET8_MASK
        self.offset1 = offset1 & OFFSET8_MASK

    def evaluate(self, lds: LDSModel64) -> int:
        return lds.atomic_dec_src2_u64(self.vaddr, self.offset0, self.offset1)


class TestDsDecSrc2U64AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_values(self) -> None:
        self.assertEqual(DsSrc2Opcode.DS_DEC_SRC2_U64.value, "ds_dec_src2_u64")
        self.assertEqual(DsSrc2Opcode.DS_INC_SRC2_U64.value, "ds_inc_src2_u64")

    def test_register_operand_validation(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        self.assertEqual(v0.format_asm(), "v0")
        s0 = Register(RegKind.SGPR, 4, count=2)
        self.assertEqual(s0.format_asm(), "s[4:5]")

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 255, count=2)

    def test_instruction_creation_and_assembly(self) -> None:
        v1 = Register(RegKind.VGPR, 1)
        inst_simple = DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v1)
        self.assertEqual(inst_simple.to_assembly(), "ds_dec_src2_u64 v1")

        inst_offsets = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=v1,
            offset0=2,
            offset1=4,
            gds=True,
        )
        self.assertEqual(inst_offsets.to_assembly(), "ds_dec_src2_u64 v1 offset0:2 offset1:4 gds")

    def test_instruction_validation_errors(self) -> None:
        s0 = Register(RegKind.SGPR, 0)
        v_pair = Register(RegKind.VGPR, 0, count=2)
        v0 = Register(RegKind.VGPR, 0)

        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=s0)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v_pair)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset0=-1)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset0=256)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset1=-5)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset1=300)

    def test_typed_expression_evaluation(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        vaddr = 0x200
        idx_a = 0x200 >> 3
        idx_b = (0x200 + 8) >> 3

        lds.write_qword(idx_a, 10)
        lds.write_qword(idx_b, 20)

        expr = DsDecSrc2Expr(vaddr=vaddr, offset0=0, offset1=1)
        res = expr.evaluate(lds)
        self.assertEqual(res, 9)
        self.assertEqual(lds.read_qword(idx_a), 9)


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
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


class ScalarLDS64Interpreter:
    """Deterministic scalar interpreter executing dual-source 64-bit LDS atomic instructions."""

    def __init__(self, lds: LDSModel64, rf: ScalarRegisterFile | None = None) -> None:
        self.lds = lds
        self.rf = rf if rf is not None else ScalarRegisterFile()

    def step(self, inst: DsSrc2Instruction) -> int:
        vaddr = self.rf.read_vgpr(inst.vaddr.index)
        new_val_a = self.lds.atomic_dec_src2_u64(
            vaddr=vaddr,
            offset0=inst.offset0,
            offset1=inst.offset1,
        )
        return new_val_a


class TestDsDecSrc2U64ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSModel64(capacity_qwords=512)
        self.rf = ScalarRegisterFile()
        self.interp = ScalarLDS64Interpreter(self.lds, self.rf)

    def test_scalar_step_normal_decrement(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, 50)
        self.lds.write_qword(idx_b, 100)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 49)
        self.assertEqual(self.lds.read_qword(idx_a), 49)
        self.assertEqual(self.lds.read_qword(idx_b), 100)

    def test_scalar_step_wrap_at_zero(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, 0)
        self.lds.write_qword(idx_b, 0x1_0000_0000)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 0x1_0000_0000)
        self.assertEqual(self.lds.read_qword(idx_a), 0x1_0000_0000)

    def test_scalar_step_reset_on_greater_threshold(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, 500)
        self.lds.write_qword(idx_b, 100)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 100)
        self.assertEqual(self.lds.read_qword(idx_a), 100)

    def test_scalar_step_with_offsets(self) -> None:
        vaddr = 0x100
        # offset0 = 2 -> 16 bytes -> idx (0x100 + 16) >> 3 = 34
        # offset1 = 4 -> 32 bytes -> idx (0x100 + 32) >> 3 = 36
        idx_a = (0x100 + 16) >> 3
        idx_b = (0x100 + 32) >> 3

        self.lds.write_qword(idx_a, 10)
        self.lds.write_qword(idx_b, 20)
        self.rf.write_vgpr(1, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 1),
            offset0=2,
            offset1=4,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 9)
        self.assertEqual(self.lds.read_qword(idx_a), 9)

    def test_scalar_step_non_returning_preserves_rf(self) -> None:
        vaddr = 0x40
        idx_a = 0x40 >> 3
        idx_b = (0x40 + 8) >> 3

        self.lds.write_qword(idx_a, 1)
        self.lds.write_qword(idx_b, 5)

        # Pre-seed registers
        for i in range(10):
            self.rf.write_vgpr(i, 0x1000 + i)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        self.interp.step(inst)
        # Verify VGPRs completely untouched except v0 remaining vaddr
        self.assertEqual(self.rf.read_vgpr(0), vaddr)
        for i in range(1, 10):
            self.assertEqual(self.rf.read_vgpr(i), 0x1000 + i)


if __name__ == "__main__":
    unittest.main()
